import os
os.environ["HF_HOME"] = "/home/taeyoung/nfs-mount/chi2027/.cache/huggingface"
os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import json
import random
import glob
import base64
import time
import pandas as pd
from tqdm.auto import tqdm
import sglang as sgl
from transformers import AutoProcessor

MODEL_NAME = "Qwen/Qwen3.5-9B"
OCR_DIR = "./data/3_ocr_results"
FRAME_DIR = "./data/2_frame_files"

SYSTEM_PROMPT = """You are a video frame analyzer that classifies OCR-detected text regions as either "telop" or "scene_text".

Telop: Text overlaid during post-production. Subtitles, captions, name plates, speech bubbles, decorative titles, reaction text, timestamps, channel logos, watermarks, and any graphical text composited onto the video frame. May be static or animated.

Scene text: Text physically existing in the 3D scene. Signage, clothing logos, book covers, product labels, screens within the scene, license plates, posters, handwritten notes, etc.

Key visual cues for telop:
- No perspective distortion (faces viewer regardless of camera angle)
- Consistent lighting unaffected by scene illumination
- Often has outline, drop shadow, stroke, or semi-transparent background
- Clean uniform font rendering
- May overlap scene objects without being occluded
- Designed for readability over video content

Key visual cues for scene text:
- Perspective distortion matching its surface
- Affected by scene lighting, shadows, reflections
- Surface texture visible beneath text
- May be partially occluded by other objects"""

N_SAMPLES = 100
OCR_SCORE_THRESHOLD = 0.4


def main():
    # 엔진 + 프로세서 로드
    print("🔧 SGLang 엔진 로드 중...")
    llm = sgl.Engine(
        model_path=MODEL_NAME,
        mem_fraction_static=0.8,
        context_length=16384,
        attention_backend="triton",
        disable_cuda_graph=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("✅ 엔진 로드 완료!\n")

    # 랜덤 샘플 수집
    print(f"🎲 랜덤 {N_SAMPLES}개 수집 중...")
    parquets = glob.glob(os.path.join(OCR_DIR, "*", "*.parquet"))
    random.shuffle(parquets)

    samples = []
    for pq in parquets:
        if len(samples) >= N_SAMPLES:
            break
        df = pd.read_parquet(pq, columns=["frame_num", "ocr_texts", "ocr_scores", "ocr_xywha"])
        has_text = df[df["ocr_texts"].apply(lambda x: len(json.loads(x)) > 0)]
        if has_text.empty:
            continue

        rel = os.path.relpath(pq, OCR_DIR)
        channel_name, pq_file = os.path.split(rel)
        video_name = pq_file[:-8]

        row = has_text.sample(1).iloc[0]
        frame_num = row["frame_num"]
        image_path = os.path.join(FRAME_DIR, channel_name, video_name, f"frame_{frame_num:08d}.jpg")
        if os.path.exists(image_path):
            samples.append((image_path, row["ocr_texts"], row["ocr_scores"], row["ocr_xywha"]))

    print(f"✅ {len(samples)}개 수집 완료\n")

    # 분류 실행
    total_telop = 0
    total_scene = 0
    total_unknown = 0
    total_regions = 0
    total_skipped = 0
    times = []
    errors = 0

    for image_path, ocr_texts_raw, ocr_scores_raw, ocr_xywha_raw in (pbar := tqdm(samples, desc="VLM 분류")):
        parts = image_path.replace("\\", "/").split("/")
        pbar.set_postfix_str(f"{parts[-3]}/{parts[-2]}", refresh=True)

        ocr_texts = json.loads(ocr_texts_raw)
        ocr_scores = json.loads(ocr_scores_raw)
        ocr_xywha = json.loads(ocr_xywha_raw)

        # score 필터링
        filtered = [(t, s, b) for t, s, b in zip(ocr_texts, ocr_scores, ocr_xywha) if s >= OCR_SCORE_THRESHOLD]
        n_skipped = len(ocr_texts) - len(filtered)
        total_skipped += n_skipped

        if filtered:
            ocr_texts, ocr_scores, ocr_xywha = zip(*filtered)
            ocr_texts, ocr_scores, ocr_xywha = list(ocr_texts), list(ocr_scores), list(ocr_xywha)
        else:
            times.append(0)
            continue

        regions = []
        for i, (text, bbox) in enumerate(zip(ocr_texts, ocr_xywha)):
            cx, cy, w, h, angle = bbox
            regions.append({
                "id": i, "text": text,
                "cx": round(cx), "cy": round(cy),
                "w": round(w), "h": round(h),
                "angle": round(angle, 1)
            })

        user_text = f"""OCR-detected text regions in this frame:
{json.dumps(regions, ensure_ascii=False)}

Classify each region. Respond ONLY with a JSON array:
[{{"id": 0, "classification": "telop"}}, {{"id": 1, "classification": "scene_text"}}, ...]"""

        # chat template 적용
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text}
                ]
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        t0 = time.time()
        try:
            output = llm.generate(
                prompt=prompt,
                sampling_params={"temperature": 0.1, "max_new_tokens": 2048},
                image_data=image_path,
            )
            elapsed = time.time() - t0
            times.append(elapsed)

            raw = output["text"].strip()
            cleaned = raw.removeprefix("```json").removesuffix("```").strip()
            classifications = json.loads(cleaned)

            for item in classifications:
                cls = item.get("classification", "unknown")
                if cls == "telop":
                    total_telop += 1
                elif cls == "scene_text":
                    total_scene += 1
                else:
                    total_unknown += 1
            total_regions += len(ocr_texts)

        except Exception as e:
            elapsed = time.time() - t0
            times.append(elapsed)
            errors += 1
            total_regions += len(ocr_texts)
            total_unknown += len(ocr_texts)
            if errors <= 3:
                print(f"\n❌ 에러: {type(e).__name__}: {e}")

    # 엔진 종료
    llm.shutdown()

    # 최종 결과
    avg_time = sum(times) / len(times) if times else 0
    print("\n" + "=" * 60)
    print(f"📊 {len(samples)}개 프레임 벤치마크 결과")
    print("=" * 60)
    print(f"  총 소요시간: {sum(times):.1f}초 ({sum(times)/60:.1f}분)")
    print(f"  프레임당 평균: {avg_time:.2f}초")
    print(f"  총 텍스트 영역: {total_regions}개 (score<{OCR_SCORE_THRESHOLD} 필터링: {total_skipped}개)")
    print(f"  🟢 텔롭: {total_telop}개 ({total_telop/max(total_regions,1)*100:.1f}%)")
    print(f"  🔴 비텔롭: {total_scene}개 ({total_scene/max(total_regions,1)*100:.1f}%)")
    print(f"  ❓ unknown/에러: {total_unknown}개")
    print(f"  에러 프레임: {errors}개")
    print(f"\n  📈 1분 영상(10fps=600프레임) 예상: ~{600*avg_time/60:.0f}분")
    print(f"  📈 10분 영상(6000프레임) 예상: ~{6000*avg_time/3600:.1f}시간")
    print("=" * 60)


if __name__ == "__main__":
    main()