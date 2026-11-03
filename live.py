"""リアルタイム文字起こし(WebSocket /ws)

ブラウザ → 16kHz Int16 PCM を約100msごとに送信
サーバー → 発話中: 高速モデル(sherpa-onnx ReazonSpeech)で約0.7秒ごとに途中結果(partial)
           発話終了(無音検出): Whisperで確定(final)、音声とテキストをDBに保存
"""
import asyncio, os, time, uuid, wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import WebSocket

SR = 16000
RMS_THR = float(os.getenv("VAD_RMS", "0.012"))            # 発話判定の音量しきい値(環境に合わせて調整)
SIL_SEC = float(os.getenv("VAD_SILENCE_SEC", "0.8"))      # この秒数の無音で発話終了
MAX_SEC = float(os.getenv("MAX_UTT_SEC", "25"))           # 長すぎる発話は強制確定(学習は30秒まで)
MIN_SEC = float(os.getenv("MIN_UTT_SEC", "0.4"))          # これより短い音は捨てる
PARTIAL_EVERY = float(os.getenv("PARTIAL_INTERVAL_SEC", "0.7"))
PREROLL_CHUNKS = 3                                         # 発話の頭切れ防止(約0.3秒)


def load_recognizer():
    d = Path(os.getenv("SHERPA_MODEL_DIR", "data/sherpa-ja"))

    def find(prefix, int8):
        files = sorted(d.glob(f"{prefix}*.onnx"))
        pick = [f for f in files if f.name.endswith(".int8.onnx") == int8]
        return str((pick or files or [None])[0]) if (pick or files) else None

    enc, dec, joi = find("encoder", True), find("decoder", False), find("joiner", True)
    tok = d / "tokens.txt"
    if not (enc and dec and joi and tok.exists()):
        print(f"[live] sherpa-onnx モデルが {d} に見つかりません。途中結果なし(確定のみ)で動きます。")
        return None
    import sherpa_onnx
    print(f"[live] sherpa-onnx model: {d}")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=enc, decoder=dec, joiner=joi, tokens=str(tok),
        num_threads=int(os.getenv("SHERPA_THREADS", "2")),
        sample_rate=SR, feature_dim=80, decoding_method="greedy_search",
    )


def decode_partial(rec, audio: np.ndarray) -> str:
    s = rec.create_stream()
    s.accept_waveform(SR, audio)
    rec.decode_stream(s)
    return s.result.text.strip()


def save_wav(path: Path, audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def attach(app, run_whisper, insert_row, audio_dir: Path, model_name: str):
    rec = load_recognizer()
    partial_pool = ThreadPoolExecutor(int(os.getenv("PARTIAL_WORKERS", "4")))
    final_pool = ThreadPoolExecutor(int(os.getenv("FINAL_WORKERS", "2")))

    class Session:
        def __init__(self, ws: WebSocket):
            self.ws = ws
            self.loop = asyncio.get_running_loop()
            self.pre = deque(maxlen=PREROLL_CHUNKS)
            self.buf, self.n, self.sil = [], 0, 0
            self.speech, self.busy, self.utt = False, False, 0
            self.last_partial = 0.0
            self.tasks = set()
            self.sid = uuid.uuid4().hex  # 1回の通話(接続)=1セッション

        async def send(self, obj):
            try:
                await self.ws.send_json(obj)
            except Exception:
                pass

        def spawn(self, coro):
            t = asyncio.create_task(coro)
            self.tasks.add(t)
            t.add_done_callback(self.tasks.discard)

        async def feed(self, data: bytes):
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if x.size == 0:
                return
            loud = float(np.sqrt(np.mean(x * x))) > RMS_THR
            if not self.speech:
                if not loud:
                    self.pre.append(x)
                    return
                self.speech = True
                self.buf = list(self.pre) + [x]
                self.n = sum(c.size for c in self.buf)
                self.sil = 0
                self.pre.clear()
            else:
                self.buf.append(x)
                self.n += x.size
                self.sil = 0 if loud else self.sil + x.size

            if self.sil >= SIL_SEC * SR or self.n >= MAX_SEC * SR:
                await self.finish()
            elif rec and not self.busy and time.monotonic() - self.last_partial >= PARTIAL_EVERY:
                self.spawn(self.partial(np.concatenate(self.buf), self.utt))

        async def partial(self, audio, utt):
            self.busy = True
            try:
                text = await self.loop.run_in_executor(partial_pool, decode_partial, rec, audio)
                if text and self.speech and utt == self.utt:  # 確定後に遅れて届いた途中結果は捨てる
                    await self.send({"type": "partial", "text": text})
            except Exception as e:
                print("[live] partial error:", e)
            finally:
                self.busy = False
                self.last_partial = time.monotonic()

        async def finish(self):
            audio = np.concatenate(self.buf)
            voiced = (audio.size - self.sil) / SR
            self.buf, self.n, self.sil, self.speech = [], 0, 0, False
            self.utt += 1
            self.pre.clear()
            if voiced < MIN_SEC:
                await self.send({"type": "discard"})
                return
            self.spawn(self.final(audio, self.utt))

        async def final(self, audio, seq):
            uid = uuid.uuid4().hex
            path = audio_dir / f"{uid}.wav"
            try:
                save_wav(path, audio)
                text = await self.loop.run_in_executor(final_pool, run_whisper, str(path))
                if not text:
                    path.unlink(missing_ok=True)
                    await self.send({"type": "discard"})
                    return
                insert_row(uid, path.name, text, self.sid, seq)
                await self.send({"type": "final", "id": uid, "text": text, "model": model_name, "seq": seq})
            except Exception as e:
                print("[live] final error:", e)
                await self.send({"type": "error", "message": str(e)})

        async def flush(self):
            if self.speech and self.n >= MIN_SEC * SR:
                await self.finish()
            elif self.speech:
                self.buf, self.n, self.sil, self.speech = [], 0, 0, False
                self.utt += 1
            if self.tasks:
                await asyncio.gather(*list(self.tasks), return_exceptions=True)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        sess = Session(ws)
        await sess.send({"type": "info", "partial": rec is not None, "model": model_name, "session": sess.sid})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await sess.feed(msg["bytes"])
                elif msg.get("text") == "flush":
                    await sess.flush()
                    await sess.send({"type": "flushed", "session": sess.sid})
        finally:
            await sess.flush()
