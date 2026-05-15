"""train+eval set 의 텔롭 텍스트에서 max char length 와 unique vocab size 를 구함.

- per-channel train/eval split (SEED=42, EVAL_PER_CHANNEL=5)
- 채널 서브샘플 안 함: 전체 채널 사용

p99 같은 통계 없이 max 만 출력.

실행:
    cd /home/taeyoung/nfs-mount/chi2027
    python scan_char_vocab.py
"""

import os
import json
import random

POS_DIR = "./data/8_telop_position"
EVAL_PER_CHANNEL = 5
SEED = 42

# 1. 채널별 json path 수집
all_channels = []
ch_paths = {}
for channel in sorted(os.listdir(POS_DIR)):
    ch_dir = os.path.join(POS_DIR, channel)
    if not os.path.isdir(ch_dir):
        continue
    all_channels.append(channel)
    ch_paths[channel] = sorted(
        os.path.join(ch_dir, f)
        for f in sorted(os.listdir(ch_dir))
        if f.endswith(".json")
    )
print(f"전체 채널: {len(all_channels)}")

# 2. per-channel train/eval split (notebook 과 동일)
rng = random.Random(SEED)
train_paths = []
eval_paths = []
for ch in all_channels:
    paths = list(ch_paths[ch])
    rng.shuffle(paths)
    n_eval = min(EVAL_PER_CHANNEL, len(paths))
    eval_paths.extend(paths[:n_eval])
    train_paths.extend(paths[n_eval:])

print(f"전체 채널 사용  train: {len(train_paths):,}  eval: {len(eval_paths):,}")

# 4. train ∪ eval 의 모든 텔롭 텍스트 스캔
max_len = 0
max_len_text = ""
max_len_file = ""
vocab = set()
total_insts = 0

all_paths = train_paths + eval_paths
for i, path in enumerate(all_paths):
    if (i + 1) % 1000 == 0:
        print(f"  스캔 {i+1:,}/{len(all_paths):,}", end="\r")
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n  ⚠️ 읽기 실패: {path} ({e})")
        continue
    for inst in data.get("instances", []):
        text = inst.get("text", "")
        L = len(text)
        if L > max_len:
            max_len = L
            max_len_text = text
            max_len_file = path
        vocab.update(text)
        total_insts += 1

print()
print(f"총 instance: {total_insts:,}")
print()
print(f"▶ max char length (글자 단위): {max_len}")
print(f"   해당 텍스트: {max_len_text!r}")
print(f"   파일: {max_len_file}")
print()
print(f"▶ unique char vocab size: {len(vocab)}")
print(f"   (padding=0, unk=1 추가 시 CHAR_VOCAB_SIZE = {len(vocab) + 2})")
