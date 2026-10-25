
import json
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI

app = FastAPI(title="FVA Backend")

@app.get("/debug-env")
def debug_env():
    return {
        "key_present": bool(
            os.environ.get("OPENAI_API_KEY", "").strip()
        )
    }
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured"
        )
    return OpenAI(api_key=key)


@app.get("/")
def home():
    return {"status": "online", "app": "FVA Backend"}


@app.get("/health")
def health():
    return {"status": "ok"}


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
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="audio_files must be a non-empty JSON list"
        )

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty recording")

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Recording too large")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp:
            temp.write(content)
            temp_path = temp.name

        with open(temp_path, "rb") as audio_input:
            transcription = client.audio.transcriptions.create(
                model="gpt-transcribe",
                file=audio_input
            )

        transcript = transcription.text.strip()
        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech recognized"
            )

        # AI chooses based on the meaning of the spoken command
        # and the user's selected audio filenames.
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=(
                "You are FVA, a semantic audio selector. "
                "Understand Hindi and Hinglish commands. "
                "Choose the single most relevant audio from the "
                "provided list based on the user's intended meaning, "
                "not just exact keyword matches. Do not invent files. "
                "Return JSON only with selected_index (zero-based "
                "integer) and message (short string). If no file fits, "
                "use selected_index -1."
            ),
            input=json.dumps({
                "spoken_command": transcript,
                "available_audio_files": [
                    {"index": i, "name": str(item.get("name", ""))}
                    for i, item in enumerate(files)
                    if isinstance(item, dict)
                ]
            }, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "audio_selection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_index": {"type": "integer"},
                            "message": {"type": "string"}
                        },
                        "required": ["selected_index", "message"],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = json.loads(response.output_text)
        index = result["selected_index"]

        if index < -1 or index >= len(files):
            index = -1

        return {
            "transcript": transcript,
            "selected_index": index,
            "message": result["message"]
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI backend error: {str(exc)[:300]}"
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
