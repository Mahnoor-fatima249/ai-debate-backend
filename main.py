import os
import requests
from fastapi import FastAPI, UploadFile, File
import whisper
from dotenv import load_dotenv

# ==========================
# ENV LOAD
# ==========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found")

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI()

# ==========================
# WHISPER MODEL
# ==========================
print("Loading Whisper Model...")
model = whisper.load_model("base", device="cpu")

# ==========================
# CLEAN TEXT
# ==========================
def clean_text(text):
    return " ".join(text.strip().split())

# ==========================
# AI FUNCTION (FINAL FIXED)
# ==========================
def get_ai_response(user_text):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a STRICT DEBATE AI. "
                    "Never analyze grammar or correctness. "
                    "Never explain sentences. "
                    "Never act like a teacher. "
                    "Respond ONLY with 1 short sentence counter-argument."
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            return f"AI Error: {response.text}"

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Exception: {str(e)}"

# ==========================
# ROUTES
# ==========================
@app.get("/")
def home():
    return {"status": "AI Debate Backend Running"}

@app.post("/debate-step")
async def debate_step(audio: UploadFile = File(...)):
    audio_file = "temp_audio.wav"

    try:
        # Save file
        with open(audio_file, "wb") as f:
            f.write(await audio.read())

        print("File saved. Now transcribing...")

        # Whisper
        result = model.transcribe(audio_file, fp16=False)
        user_text = clean_text(result["text"])

        print("User said:", user_text)

        if len(user_text.split()) < 3:
            return {"error": "Speech too short"}

        # FORCE DEBATE MODE
        user_text = "Debate topic: " + user_text

        # AI RESPONSE
        print("Generating AI response...")
        ai_reply = get_ai_response(user_text)

        print("AI RESPONSE:", ai_reply)

        return {
            "user_said": user_text,
            "ai_response": ai_reply
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)