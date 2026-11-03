"""CER(文字誤り率)を計測する。

台本と比べる(通話1回分。区切りごとの文をつなげて台本全体と比べるので、行ごとの訂正は不要):
    python scripts/cer.py --ref scripts/daihon.txt
    python scripts/cer.py --ref scripts/daihon.txt --models small,medium,large-v3-turbo --sherpa

画面で訂正して「学習に使う」にした区切りと比べる:
    python scripts/cer.py --models small,medium

--models を付けると、保存済みの音声を各モデルで認識し直して比べる(server.py と同じ設定)。
比べる前に、全角半角をそろえ(NFKC)、句読点と空白を除く。漢数字と算用数字の違いは誤りに数える。
"""
import argparse, sqlite3, sys, time, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # live.py を読み込むため
DATA = Path("data")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(c for c in s if not unicodedata.category(c)[0] in "PZC")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    return edit_distance(r, h) / max(len(r), 1)


def load(ref_path):
    con = sqlite3.connect(DATA / "asr.db")
    con.row_factory = sqlite3.Row
    if ref_path:
        sid = con.execute("SELECT session_id FROM utterances WHERE session_id IS NOT NULL"
                          " ORDER BY created DESC LIMIT 1").fetchone()
        if not sid:
            raise SystemExit("通話の記録がありません。/live.html で台本を読み上げてください。")
        rows = con.execute("SELECT * FROM utterances WHERE session_id=? ORDER BY seq",
                           (sid[0],)).fetchall()
        lines = [l.strip() for l in Path(ref_path).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        return rows, "".join(lines)
    rows = con.execute("SELECT * FROM utterances WHERE approved=1 ORDER BY created").fetchall()
    if not rows:
        raise SystemExit("「学習に使う」にした区切りがありません。")
    return rows, None


def glossary_prompt():
    p = Path("glossary.txt")
    if not p.exists():
        return None
    words = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    return "、".join(words) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="台本ファイル。省略時は訂正済みの区切りと比べる")
    ap.add_argument("--models", default="", help="認識し直すWhisperモデル(カンマ区切り)")
    ap.add_argument("--sherpa", action="store_true", help="途中表示用モデル(ReazonSpeech)でも認識する")
    ap.add_argument("--show", action="store_true", help="各モデルの認識結果も表示する")
    args = ap.parse_args()

    rows, ref_all = load(args.ref)
    paths = [str(DATA / "audio" / r["audio"]) for r in rows]
    refs = [r["text"] for r in rows]

    from faster_whisper.audio import decode_audio
    audios = [decode_audio(p, sampling_rate=16000) for p in paths]
    audio_sec = sum(a.size for a in audios) / 16000

    def score(hyps):
        if args.show:
            print("\n".join(hyps), end="\n\n")
        if ref_all is not None:
            return cer(ref_all, "".join(hyps))
        total = sum(len(norm(r)) for r in refs)
        return sum(edit_distance(norm(r), norm(h)) for r, h in zip(refs, hyps)) / max(total, 1)

    results = [("保存時の結果(" + (rows[0]["model"] or "?") + ")", score([r["asr_text"] for r in rows]), None)]

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        from faster_whisper import WhisperModel
        model = WhisperModel(name, device="cpu", compute_type="int8")
        t = time.perf_counter()
        hyps = []
        for a in audios:
            segs, _ = model.transcribe(a, language="ja", beam_size=5, vad_filter=True,
                                       initial_prompt=glossary_prompt())
            hyps.append("".join(s.text for s in segs).strip())
        results.append((f"Whisper {name}", score(hyps), time.perf_counter() - t))
        del model

    if args.sherpa:
        from live import load_recognizer, decode_partial
        rec = load_recognizer()
        if rec is None:
            raise SystemExit("途中表示用モデルがありません。python scripts/fetch_sherpa.py を実行してください。")
        t = time.perf_counter()
        hyps = [decode_partial(rec, a) for a in audios]
        results.append(("ReazonSpeech(途中表示用)", score(hyps), time.perf_counter() - t))

    print(f"区切り {len(rows)} 個 / 音声 {audio_sec:.1f} 秒 / 正解 {len(norm(ref_all or ''.join(refs)))} 文字\n")
    print("| モデル | CER | 処理時間(音声1秒あたり) |")
    print("|---|---|---|")
    for name, c, sec in results:
        speed = f"{sec / audio_sec:.2f} 秒" if sec is not None else "-"
        print(f"| {name} | {c * 100:.1f}% | {speed} |")


if __name__ == "__main__":
    main()
