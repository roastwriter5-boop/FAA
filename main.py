
import json
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI

app = FastAPI(title="FVA Backend")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
SELECTION_MODEL = "openai/gpt-oss-20b"


@app.get("/")
def home():
    return {"status": "online", "app": "FVA Backend"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug-env")
def debug_env():
    return {
        "groq_key_present": bool(
            os.getenv("GROQ_API_KEY", "").strip()
        )
    }


def get_client():
    key = os.getenv("GROQ_API_KEY", "").strip()

    if not key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured in Render"
        )

    return OpenAI(
        api_key=key,
        base_url=GROQ_BASE_URL,
        timeout=60.0,
        max_retries=1
    )


@app.post("/voice-command")
async def voice_command(
    audio: UploadFile = File(...),
    audio_files: str = Form(...)
):
    client = get_client()

    try:
        files = json.loads(audio_files)

        if not isinstance(files, list) or not files:
            raise ValueError("No audio files supplied")

        if not all(isinstance(item, dict) for item in files):
            raise ValueError("Invalid audio file list")

    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="audio_files must be a non-empty JSON list of objects"
        )

    content = await audio.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Empty recording"
        )

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Recording too large"
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        # Step 1: Transcribe spoken Hindi/Hinglish.
        with open(temp_path, "rb") as audio_input:
            transcription = client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_input
            )

        transcript = transcription.text.strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech recognized"
            )

        # Preserve original indexes so the Android app selects
        # the correct audio item.
        available_files = [
            {
                "index": i,
                "name": str(item.get("name", ""))
            }
            for i, item in enumerate(files)
        ]

        # Step 2: Select audio by meaning, not exact keywords.
        response = client.chat.completions.create(
            model=SELECTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are FVA, a semantic audio selector. "
                        "Understand Hindi, Hinglish, and natural "
                        "spoken commands. Select the single audio "
                        "file that best matches the user's intended "
                        "meaning, even if the command uses different "
                        "words from the filename. Use only the "
                        "provided files. Never invent a file. "
                        "If no file is a reasonable match, use "
                        "selected_index -1. Return JSON only with "
                        "selected_index (integer) and message "
                        "(short string)."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "spoken_command": transcript,
                            "available_audio_files": available_files
                        },
                        ensure_ascii=False
                    )
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        output = response.choices[0].message.content

        if not output:
            raise ValueError("AI returned an empty selection")

        result = json.loads(output)

        index = result.get("selected_index", -1)
        message = result.get("message", "")

        if isinstance(index, bool) or not isinstance(index, int):
            index = -1

        if index < -1 or index >= len(files):
            index = -1

        if not isinstance(message, str):
            message = ""

        return {
            "transcript": transcript,
            "selected_index": index,
            "message": message
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"FVA backend error: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=502,
            detail=f"AI backend error: {str(exc)[:300]}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
