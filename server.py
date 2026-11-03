"""社内完結 音声認識プロトタイプ(FastAPI + faster-whisper + SQLite)"""
import os, sqlite3, threading, time, uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel

DATA = Path(os.getenv("DATA_DIR", "data"))
AUDIO = DATA / "audio"
AUDIO.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "asr.db"
CURRENT = DATA / "current_model.txt"  # train.py が改善時に書き換える

def model_name() -> str:
    if os.getenv("ASR_MODEL"):
        return os.environ["ASR_MODEL"]
    if CURRENT.exists():
        return CURRENT.read_text().strip()
    return "large-v3-turbo"

MODEL_NAME = model_name()
print(f"[asr] loading model: {MODEL_NAME}")
model = WhisperModel(
    MODEL_NAME,
    device=os.getenv("ASR_DEVICE", "auto"),
    compute_type=os.getenv("ASR_COMPUTE", "default"),
)
model_lock = threading.Lock()

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

with db() as con:
    con.execute("""CREATE TABLE IF NOT EXISTS utterances(
        id TEXT PRIMARY KEY, created REAL, audio TEXT, model TEXT,
        asr_text TEXT, text TEXT, approved INTEGER DEFAULT 0, updated REAL,
        session_id TEXT, seq INTEGER)""")
    cols = {r["name"] for r in con.execute("PRAGMA table_info(utterances)")}
    if "session_id" not in cols:  # 旧バージョンのDBを引き継ぐ
        con.execute("ALTER TABLE utterances ADD COLUMN session_id TEXT")
    if "seq" not in cols:
        con.execute("ALTER TABLE utterances ADD COLUMN seq INTEGER")

def glossary_prompt() -> str | None:
    p = Path("glossary.txt")
    if not p.exists():
        return None
    words = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    return "、".join(words) if words else None

app = FastAPI()

def run_whisper(path: str) -> str:
    """確定用の認識(Whisper)。GPU共有のためロックで直列化。"""
    with model_lock:
        segs, _ = model.transcribe(
            path, language="ja", beam_size=5, vad_filter=True,
            initial_prompt=glossary_prompt(),
        )
        return "".join(s.text for s in segs).strip()

def insert_row(uid: str, audio_name: str, text: str, session_id=None, seq=None):
    now = time.time()
    with db() as con:
        con.execute(
            "INSERT INTO utterances(id,created,audio,model,asr_text,text,approved,updated,session_id,seq)"
            " VALUES(?,?,?,?,?,?,0,?,?,?)",
            (uid, now, audio_name, MODEL_NAME, text, text, now, session_id, seq))

@app.post("/api/transcribe")
def transcribe(file: UploadFile = File(...)):
    ext = Path(file.filename or "a.webm").suffix or ".webm"
    uid = uuid.uuid4().hex
    path = AUDIO / f"{uid}{ext}"
    path.write_bytes(file.file.read())
    text = run_whisper(str(path))
    insert_row(uid, path.name, text)
    return {"id": uid, "text": text, "model": MODEL_NAME}

@app.get("/api/items")
def items():
    with db() as con:
        rows = con.execute("SELECT * FROM utterances ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]

class Edit(BaseModel):
    text: str
    approved: bool = False

@app.put("/api/items/{uid}")
def edit(uid: str, body: Edit):
    with db() as con:
        cur = con.execute("UPDATE utterances SET text=?, approved=?, updated=? WHERE id=?",
                          (body.text, int(body.approved), time.time(), uid))
    if cur.rowcount == 0:
        raise HTTPException(404)
    return {"ok": True}

@app.delete("/api/items/{uid}")
def delete(uid: str):
    with db() as con:
        row = con.execute("SELECT audio FROM utterances WHERE id=?", (uid,)).fetchone()
        con.execute("DELETE FROM utterances WHERE id=?", (uid,))
    if row:
        (AUDIO / row["audio"]).unlink(missing_ok=True)
    return {"ok": True}

@app.get("/api/audio/{uid}")
def audio(uid: str):
    with db() as con:
        row = con.execute("SELECT audio FROM utterances WHERE id=?", (uid,)).fetchone()
    if not row:
        raise HTTPException(404)
    return FileResponse(AUDIO / row["audio"])

@app.get("/api/stats")
def stats():
    with db() as con:
        r = con.execute("SELECT COUNT(*) n, SUM(approved) a FROM utterances").fetchone()
    return {"model": MODEL_NAME, "total": r["n"], "approved": r["a"] or 0}

@app.get("/api/export", response_class=PlainTextResponse)
def export():
    import json
    with db() as con:
        rows = con.execute("SELECT id,audio,asr_text,text FROM utterances WHERE approved=1").fetchall()
    return "\n".join(json.dumps(dict(r), ensure_ascii=False) for r in rows)

# ---- 通話(セッション)単位の取得・訂正 ----
@app.get("/api/sessions")
def sessions():
    with db() as con:
        rows = con.execute("""SELECT session_id, MIN(created) AS started, COUNT(*) AS n,
            SUM(approved) AS approved,
            (SELECT text FROM utterances u2 WHERE u2.session_id=u.session_id
             ORDER BY seq LIMIT 1) AS preview
            FROM utterances u WHERE session_id IS NOT NULL
            GROUP BY session_id ORDER BY started DESC LIMIT 100""").fetchall()
    return [dict(r) for r in rows]

@app.get("/api/sessions/{sid}")
def session_items(sid: str):
    with db() as con:
        rows = con.execute("SELECT * FROM utterances WHERE session_id=? ORDER BY seq, created", (sid,)).fetchall()
    if not rows:
        raise HTTPException(404)
    return [dict(r) for r in rows]

class Line(BaseModel):
    id: str
    text: str

class SessionEdit(BaseModel):
    items: list[Line]
    approved: bool = False

@app.put("/api/sessions/{sid}")
def session_edit(sid: str, body: SessionEdit):
    now = time.time()
    with db() as con:
        for it in body.items:
            con.execute("UPDATE utterances SET text=?, approved=?, updated=? WHERE id=? AND session_id=?",
                        (it.text, int(body.approved), now, it.id, sid))
    return {"ok": True}

@app.delete("/api/sessions/{sid}")
def session_delete(sid: str):
    with db() as con:
        rows = con.execute("SELECT audio FROM utterances WHERE session_id=?", (sid,)).fetchall()
        con.execute("DELETE FROM utterances WHERE session_id=?", (sid,))
    for r in rows:
        (AUDIO / r["audio"]).unlink(missing_ok=True)
    return {"ok": True}

import live
live.attach(app, run_whisper, insert_row, AUDIO, MODEL_NAME)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
