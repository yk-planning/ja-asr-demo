# ja-asr-demo — 日本語音声認識プロトタイプ

ブラウザで話すと、自分のサーバーだけで日本語を文字起こしします。外部APIは使いません(モデルの初回ダウンロードを除く)。
訂正したデータで Whisper を再学習(LoRA)し、精度が上がったときだけモデルを差し替えられます。

- 確定用: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- 途中表示用(任意): [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) + ReazonSpeech
- サーバー: FastAPI + SQLite

## 使い方

### 1. 起動する(Docker)
```
git clone https://github.com/yk-planning/ja-asr-demo.git
cd ja-asr-demo
python scripts/fetch_sherpa.py      # 途中表示用モデル(約170MB)。不要なら省略可
docker compose up --build           # 初回はビルドとモデル取得に時間がかかります
```
ブラウザで http://localhost:8000 を開きます。止めるときは Ctrl+C、または `docker compose down`。

- 既定は CPU 用の `small` モデルです。変更例(PowerShell): `$env:ASR_MODEL="medium"; docker compose up`
- 録音・DB・モデルは `./data` に保存されます。
- 途中表示用モデルを後から入れた場合は `docker compose restart` します。

### 2. 録音して文字起こしする(/)
1. 「録音を開始」を押して話します(1回30秒以内、短い文単位がおすすめ)。
2. 文字起こしが一覧に出ます。誤りがあれば直します。
3. 学習に使うものは「学習に使う」にチェックして保存します。

### 3. 通話のように話し続ける(/live.html)
1. 「通話を始める」を押して話します。
   - 話している間は途中結果が薄い色で出ます(途中表示用モデルがある場合)。
   - 無音で区切られるたびに、Whisper の確定文が下に追記されます。
2. 「通話を終える」を押すと、通話全体が1画面に並びます。
   - 行ごとに直せます。▶で該当箇所を再生、「続けて再生」で通話の頭から通して聞けます。
3. 「この通話を学習に使う」にチェックして保存すると、通話内のすべての区切りが再学習の対象になります。

過去の通話は「これまでの通話」から開き直せます。

### 4. 専門用語を覚えさせる(再学習なし)
`glossary.txt` に社名・製品名・専門用語を1行ずつ書きます(`#` で始まる行は無視)。
認識時のヒント(Whisper の `initial_prompt`)として渡され、再起動なしで効きます。

### 5. 再学習する
承認済みデータが20件以上になったら実行します。
```
docker compose run --rm -e BASE_MODEL=openai/whisper-small -e MIN_ITEMS=10 asr python train.py
```
- Docker で学習するには、先に `$env:WITH_TRAIN="1"; docker compose build` で学習用ライブラリを入れます。
- 約10%を検証用に分け、学習前後の CER(文字誤り率)を表示します。
- 改善したときだけ `data/models/<日時>/ct2` に保存し、`data/current_model.txt` を更新します。
- サーバーを再起動すると新モデルで動きます。戻すには `data/current_model.txt` を削除して再起動します。
- 学習は GPU(VRAM 24GB 目安、`small` なら 8GB 程度)を推奨します。CPU だと非常に遅いです。

## 設定(環境変数)

| 変数 | 既定 | 意味 |
|---|---|---|
| ASR_MODEL | `small`(Docker) | Whisper のモデル。`small` / `medium` / `large-v3` / `large-v3-turbo` など |
| ASR_DEVICE | `cpu`(Docker) | `cpu` / `cuda` / `auto` |
| ASR_COMPUTE | `int8`(Docker) | 計算精度。GPU なら `float16` など |
| VAD_RMS | 0.012 | 発話と判定する音量。騒がしい場所では上げる / 拾わないなら下げる |
| VAD_SILENCE_SEC | 0.8 | この秒数の無音で発話を区切る |
| PARTIAL_INTERVAL_SEC | 0.7 | 途中結果の更新間隔 |
| MAX_UTT_SEC | 25 | 長い発話の強制確定 |
| SHERPA_MODEL_DIR | `data/sherpa-ja` | 途中表示用モデルの場所 |
| HF_HUB_OFFLINE | 未設定 | `1` にするとモデル取得後に完全オフラインで動く |

## Docker を使わずに動かす
```
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_sherpa.py                       # 任意
set ASR_MODEL=small& set ASR_DEVICE=cpu& set ASR_COMPUTE=int8   # GPU があれば省略可
uvicorn server:app --host 127.0.0.1 --port 8000
```
再学習も試す場合は追加で `pip install torch` と `pip install -r requirements-train.txt`。

## 別のPCやWebから使う
ブラウザのマイクは localhost 以外では HTTPS が必要です。

- **社内のPCから**: 社内CAや自己署名証明書を用意して起動します。
  ```
  uvicorn server:app --host 0.0.0.0 --port 8443 --ssl-keyfile key.pem --ssl-certfile cert.pem
  ```
- **Webに公開(Cloudflare Tunnel)**:
  1. Cloudflare の Zero Trust ダッシュボードで Tunnel を作成し、トークンを控える。
  2. Tunnel の Public Hostname に `asr.自社ドメイン` を追加し、接続先を `http://asr:8000` にする。
  3. `$env:TUNNEL_TOKEN="(トークン)"; docker compose -f docker-compose.web.yml up -d --build`
  4. **必ず** Cloudflare Access で、このホスト名にログイン認証を付ける。

## 制限と注意
- **このアプリにはログイン機能がありません。** 開ける人には全員の録音と文字起こしが見え、削除もできます。公開するときは必ず認証を前に置いてください。
- 音声と文字は `data/` に保存されます。アクセス権限と保存期間を決めてください。
- 確定用の Whisper はロックで直列処理です。同時に多人数で使う構成(キュー化・GPUワーカー分離・PostgreSQL化)にはなっていません。
- 無音検出は音量だけの簡易版です。誤判定が多ければ Silero VAD などに差し替えてください。
- 日本語の ReazonSpeech は非ストリーミング型なので、発話中の音声を繰り返し認識し直す「疑似ストリーミング」にしています。
- 再学習は、数十件では傾向確認までです。実用的な改善には数時間分以上のデータが目安です。同じ人・同じ内容ばかりだと評価が甘くなります。

## ライセンス
[MIT](LICENSE)。モデル(Whisper、ReazonSpeech)はそれぞれの配布元のライセンスに従います。
