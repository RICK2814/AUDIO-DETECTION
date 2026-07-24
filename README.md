# Multilingual Indian Language Identification — Live Dashboard

This is a two-part app:

- **`backend/main.py`** — a small FastAPI server that does the *real* inference by
  calling Groq's hosted **Whisper Large v3** model. Your Groq API key lives only
  here, as an environment variable — it is never sent to the browser.
- **`lid_dashboard.html`** — the dashboard UI. It uploads audio to the backend and
  displays whatever the backend actually returns. No random numbers.

## ⚠️ About your API key

You pasted a live Groq key in this chat. Treat it as compromised — **rotate it**
at https://console.groq.com/keys once you're done testing, and never paste keys
into a chat window again. Going forward, only ever put it in an environment
variable on your own machine/server.

## What's real vs. approximate here

- **Detected language** — real. Comes directly from Whisper Large v3 via Groq.
- **Confidence %** — an *approximation*, derived from Whisper's own segment-level
  log-probabilities. Whisper's API doesn't expose a calibrated probability, so
  treat this as a rough signal, not a precise number.
- **Other 8 language bars** — Whisper returns a single top-1 language, not a full
  9-way distribution. The UI does not fabricate numbers for the non-detected
  languages; it says so explicitly.
- **Odia & Konkani** — not part of Whisper's language set. If the audio is
  actually Odia or Konkani, Whisper will guess its closest known language
  instead, and the UI will flag this mismatch with a warning banner rather than
  silently showing a wrong-but-confident-looking result.

If you need genuinely accurate coverage of all 9 languages including Odia and
Konkani, you'd need a model actually trained on labeled data for those two
languages (e.g. the Wav2Vec2-XLSR-53 + MLP classifier your paper describes) —
Whisper alone can't get you there.

## Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

export GROQ_API_KEY="your-new-rotated-key-here"
uvicorn main:app --reload --port 8000
```

Leave that running, then open `lid_dashboard.html` directly in your browser
(double-click it, or serve it with `python -m http.server 8080` and visit
`http://localhost:8080/lid_dashboard.html`).

Upload a WAV/MP3 clip, hit **Predict Language**, and it will call your local
backend, which calls Groq, and shows you the real result.

## If the frontend can't reach the backend

Some browsers restrict `fetch()` from a `file://` page. If you see a
"Could not reach the inference backend" warning:

1. Confirm the backend is running: visit `http://localhost:8000/api/health` —
   it should show `{"status":"ok","key_configured":true}`.
2. Serve the HTML file over `http://` instead of opening it directly:
   `python -m http.server 8080` in the folder containing `lid_dashboard.html`,
   then open `http://localhost:8080/lid_dashboard.html`.
3. If your backend runs on a different host/port, set it before the page loads
   by adding this above the closing `</head>` tag in the HTML:
   `<script>window.LID_BACKEND_URL = "http://your-host:port";</script>`

## Deploying beyond your own machine

Don't ship the backend with the key hardcoded, and don't expose `/api/predict`
publicly without rate limiting/auth — anyone who finds the URL could run up
your Groq bill. For a real deployment, put this behind your own auth layer.
