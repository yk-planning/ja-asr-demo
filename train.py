"""承認済みの訂正データで Whisper を LoRA ファインチューニングし、
CER(文字誤り率)が改善した場合のみ faster-whisper 形式に変換して差し替える。

使い方:  python train.py            (環境変数 BASE_MODEL で学習元を変更可)
"""
import hashlib, os, random, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

import jiwer, torch
from faster_whisper.audio import decode_audio
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DATA = Path(os.getenv("DATA_DIR", "data"))
BASE = os.getenv("BASE_MODEL", "openai/whisper-large-v3-turbo")  # 試すだけなら openai/whisper-small
EPOCHS = int(os.getenv("EPOCHS", 3))
LR = float(os.getenv("LR", 1e-4))
BS, ACCUM = int(os.getenv("BATCH", 2)), int(os.getenv("ACCUM", 4))
MIN_ITEMS = int(os.getenv("MIN_ITEMS", 20))
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else None

# ---- データ読み込み ----
con = sqlite3.connect(DATA / "asr.db"); con.row_factory = sqlite3.Row
rows = con.execute("SELECT id,audio,text FROM utterances WHERE approved=1 AND text!=''").fetchall()
if len(rows) < MIN_ITEMS:
    sys.exit(f"承認済みデータが {len(rows)} 件です。{MIN_ITEMS} 件以上集めてください(MIN_ITEMS で変更可)。")

def is_val(uid):  # IDのハッシュで固定的に約10%を検証用へ
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 10 == 0

proc = WhisperProcessor.from_pretrained(BASE, language="japanese", task="transcribe")
proc.tokenizer.set_prefix_tokens(language="japanese", task="transcribe")

def load(r):
    audio = decode_audio(str(DATA / "audio" / r["audio"]), sampling_rate=16000)[: 16000 * 30]  # 30秒まで
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
    return {"feats": feats, "ids": proc.tokenizer(r["text"]).input_ids, "text": r["text"]}

data = [(is_val(r["id"]), load(r)) for r in rows]
train = [d for v, d in data if not v]; val = [d for v, d in data if v]
if not val:  # 少量時の保険
    val = train[:max(1, len(train) // 10)]; train = train[len(val):]
print(f"train={len(train)} val={len(val)} base={BASE} device={dev}")

# ---- 評価 ----
@torch.no_grad()
def cer(model):
    model.eval(); hyp, ref = [], []
    for i in range(0, len(val), 4):
        b = val[i:i + 4]
        x = torch.stack([d["feats"] for d in b]).to(dev)
        with torch.autocast(dev, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model.generate(input_features=x, language="ja", task="transcribe", num_beams=5, max_new_tokens=200)
        hyp += [t.strip() for t in proc.batch_decode(out, skip_special_tokens=True)]
        ref += [d["text"].strip() for d in b]
    return jiwer.cer(ref, hyp)

model = WhisperForConditionalGeneration.from_pretrained(BASE).to(dev)
before = cer(model); print(f"CER before: {before:.4f}")

# ---- LoRA 学習 ----
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"]))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
start_id = model.config.decoder_start_token_id
random.seed(0)
for ep in range(EPOCHS):
    model.train(); random.shuffle(train); tot = 0
    for step, i in enumerate(range(0, len(train), BS)):
        b = train[i:i + BS]
        x = torch.stack([d["feats"] for d in b]).to(dev)
        L = max(len(d["ids"]) for d in b)
        lab = torch.full((len(b), L), -100)
        for j, d in enumerate(b): lab[j, :len(d["ids"])] = torch.tensor(d["ids"])
        if (lab[:, 0] == start_id).all(): lab = lab[:, 1:]  # 先頭のSOTは内部で付与される
        with torch.autocast(dev, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(input_features=x, labels=lab.to(dev)).loss
        (loss / ACCUM).backward(); tot += loss.item()
        if (step + 1) % ACCUM == 0:
            opt.step(); opt.zero_grad()
    opt.step(); opt.zero_grad()
    print(f"epoch {ep + 1}/{EPOCHS} loss={tot / max(1, step + 1):.4f}")

model = model.merge_and_unload(); model.config.use_cache = True
after = cer(model); print(f"CER after:  {after:.4f}  (before {before:.4f})")

# ---- 改善時のみ差し替え ----
if after >= before:
    sys.exit("改善しなかったため、モデルは差し替えません。データを増やすか、訂正内容を見直してください。")

ver = time.strftime("%Y%m%d-%H%M%S")
hf_dir, ct2_dir = DATA / "models" / ver / "hf", DATA / "models" / ver / "ct2"
model.save_pretrained(hf_dir); proc.save_pretrained(hf_dir)
subprocess.run(["ct2-transformers-converter", "--model", str(hf_dir), "--output_dir", str(ct2_dir),
                "--copy_files", "tokenizer.json", "preprocessor_config.json",
                "--quantization", "float16"], check=True)
(DATA / "current_model.txt").write_text(str(ct2_dir.resolve()))
shutil.rmtree(hf_dir, ignore_errors=True)
print(f"新モデルを {ct2_dir} に保存しました。サーバーを再起動すると反映されます。")
print("元に戻すには data/current_model.txt を削除(または以前のパスに書き換え)して再起動してください。")
