"""
Backend for the Multilingual Indian Language Identification dashboard.

Runs real inference via Groq's hosted Whisper Large v3 model. The API key
is read from an environment variable (GROQ_API_KEY) — it is never sent to
or embedded in the browser/frontend code.

Run:
    export GROQ_API_KEY="your-key-here"
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

The frontend (lid_dashboard.html) calls POST /api/predict on this server.
"""

import os
import time
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

app = FastAPI(title="LID Inference Backend")

# Allow the static HTML dashboard (opened via file:// or any local dev server) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Target 9-language set for this research project, with metadata used to
# render the UI. Whisper's ISO-639-1 code -> our language entry.
# NOTE: Odia ("or") and Konkani ("kok") are NOT part of Whisper's supported
# language set, so they can never be a *direct* Whisper prediction. If the
# audio is actually Odia/Konkani, Whisper will guess the closest language
# it does know (often Hindi, Bengali, or Marathi) — we surface that
# mismatch honestly to the user rather than hiding it.
# ---------------------------------------------------------------------------
TARGET_LANGUAGES = {
    "bn": {"en": "Bengali", "native": "বাংলা", "family": "Indo-Aryan", "in_target_set": True},
    "gu": {"en": "Gujarati", "native": "ગુજરાતી", "family": "Indo-Aryan", "in_target_set": True},
    "kn": {"en": "Kannada", "native": "ಕನ್ನಡ", "family": "Dravidian", "in_target_set": True},
    "ml": {"en": "Malayalam", "native": "മലയാളം", "family": "Dravidian", "in_target_set": True},
    "mr": {"en": "Marathi", "native": "मराठी", "family": "Indo-Aryan", "in_target_set": True},
    "ta": {"en": "Tamil", "native": "தமிழ்", "family": "Dravidian", "in_target_set": True},
    "te": {"en": "Telugu", "native": "తెలుగు", "family": "Dravidian", "in_target_set": True},
}
# Not directly detectable by Whisper — listed so the frontend can still show them as "supported by project" chips.
UNSUPPORTED_BY_WHISPER = {
    "or": {"en": "Odia", "native": "ଓଡ଼ିଆ", "family": "Indo-Aryan", "in_target_set": True},
    "kok": {"en": "Konkani", "native": "कोंकणी", "family": "Indo-Aryan", "in_target_set": True},
}


@app.get("/api/health")
def health():
    return {"status": "ok", "key_configured": bool(GROQ_API_KEY)}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Set it as an environment "
                   "variable before starting the backend — see README.md.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav")},
                data={
                    "model": "whisper-large-v3",
                    "response_format": "verbose_json",
                    # no "language" param -> ask Whisper to auto-detect
                },
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Could not reach Groq API: {e}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Groq API error: {resp.text}")

    data = resp.json()
    detected_code = (data.get("language") or "").lower()
    transcript = data.get("text", "").strip()

    # Rough confidence proxy: Whisper doesn't return a class probability,
    # so we derive an approximate confidence signal from segment-level
    # avg_logprob / no_speech_prob when available. This is disclosed to
    # the frontend as an approximation, not a calibrated probability.
    segments = data.get("segments") or []
    if segments:
        avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
        no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
        # squash avg_logprob (~[-1, 0]) into a 0-100 confidence-ish scale
        approx_confidence = max(5.0, min(99.0, (1 + avg_logprob) * 100 * (1 - no_speech)))
    else:
        approx_confidence = None

    in_target = detected_code in TARGET_LANGUAGES
    lang_info = TARGET_LANGUAGES.get(detected_code)

    result = {
        "detected_code": detected_code,
        "in_target_set": in_target,
        "language": lang_info,  # None if Whisper detected something outside our 9-language project scope
        "transcript_preview": transcript[:200],
        "approx_confidence_pct": round(approx_confidence, 2) if approx_confidence is not None else None,
        "inference_time_ms": elapsed_ms,
        "model": "whisper-large-v3 (Groq)",
        "note": (
            "Odia and Konkani are not part of Whisper's language set — if your audio is in "
            "one of those languages, Whisper will report its closest guess among the "
            "languages it knows, which will be inaccurate for this project's purposes."
            if not in_target else None
        ),
    }
    return JSONResponse(result)
