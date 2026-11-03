"""途中表示用モデル(ReazonSpeech)を data/sherpa-ja に取得する。live.py が使う4ファイルだけを取り出す。"""
import sys, tarfile, urllib.request
from pathlib import Path

URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2"
KEEP = {"encoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"}

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sherpa-ja")
out.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(URL) as r, tarfile.open(fileobj=r, mode="r|bz2") as t:
    for m in t:
        name = m.name.rsplit("/", 1)[-1]
        if m.isfile() and name in KEEP:
            (out / name).write_bytes(t.extractfile(m).read())
            print("saved", name)
missing = KEEP - {p.name for p in out.iterdir()}
if missing:
    sys.exit(f"見つからないファイル: {missing}")
