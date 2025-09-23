# -*- coding: utf-8 -*-
import os, json
from tempfile import NamedTemporaryFile

import gradio as gr
import whisper
from gtts import gTTS
from groq import Groq

# ----------------------------
# Groq Client (kept inline)
# ----------------------------
client = Groq(
    api_key="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"  # keep your key here if you prefer
)

# Whisper model (you can switch to "tiny"/"small" for faster load)
asr_model = whisper.load_model("base")

# Simple language map
LANG_MAP = {
    "English": ("en", "English"),
    "Hindi": ("hi", "Hindi"),
}

# ----------------------------
# Tiny disk persistence
# ----------------------------
MEMORY_PATH = "memory.json"
HISTORY_LIMIT = 12  # keep last N turns per user to avoid huge prompts

def _load_all_mem():
    if not os.path.exists(MEMORY_PATH):
        return {}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_all_mem(mem):
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # keep the pipeline resilient

def load_user_mem(user_id: str):
    all_mem = _load_all_mem()
    return all_mem.get(user_id, {"preferred_language": "en", "history": []})

def save_user_mem(user_id: str, user_mem: dict):
    all_mem = _load_all_mem()
    all_mem[user_id] = user_mem
    _save_all_mem(all_mem)

# ----------------------------
# Core steps
# ----------------------------
def speech_to_text(audio_path: str, lang_code: str) -> str:
    """Transcribe with Whisper; force selected language for consistency."""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        out = asr_model.transcribe(audio_path, language=lang_code)
        return (out.get("text") or "").strip()
    except Exception as e:
        return f"[ASR error] {e}"

def generate_response(text: str, lang_code: str, lang_name: str, user_history):
    """
    Ask Groq LLM to reply strictly in the selected language, with short memory.
    user_history is a list of {"role": "...", "content": "..."} we append.
    """
    text = (text or "").strip()
    if not text:
        return "I didn't catch that. Please try again." if lang_code == "en" else "मैं समझ नहीं पाया। कृपया फिर से बोलें।"

    # Build messages: system + clipped history + new user message
    messages = [
        {
            "role": "system",
            "content": (
                f"Reply ONLY in {lang_name}. "
                f"Keep answers concise and helpful. Use the user's prior context if present."
            ),
        }
    ]
    # clip history to last HISTORY_LIMIT turns
    if user_history:
        messages.extend(user_history[-HISTORY_LIMIT:])
    messages.append({"role": "user", "content": text})

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # swap to "llama-3.3-70b-versatile" if you want bigger
            messages=messages,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[Groq error] {e}" if lang_code == "en" else f"[Groq त्रुटि] {e}"

def text_to_speech(text: str, lang_code: str):
    """Convert reply to speech; if TTS fails, return None for audio."""
    try:
        tts = gTTS(
            text or ("Sorry, I couldn't generate a response." if lang_code == "en" else "क्षमा करें, मैं उत्तर नहीं दे सका।"),
            lang=lang_code
        )
        f = NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(f.name)
        return f.name
    except Exception:
        return None

# ----------------------------
# Pipeline with memory
# ----------------------------
def chatbot_pipeline(audio_path, language_choice, user_id, persist_memory):
    """
    language_choice: "English" or "Hindi"
    user_id: identifier for storing/retrieving memory (e.g., phone/roll/email)
    persist_memory: bool toggle to save conversation + preferences
    """
    # Normalize user_id
    user_id = (user_id or "guest").strip() or "guest"
    lang_code, lang_name = LANG_MAP.get(language_choice, ("en", "English"))

    # Load user's memory (history + preferred language)
    user_mem = load_user_mem(user_id)
    history = user_mem.get("history", [])
    # Optional: if no language explicitly chosen, fall back to stored pref
    if language_choice not in LANG_MAP and user_mem.get("preferred_language"):
        lang_code = user_mem["preferred_language"]
        lang_name = "Hindi" if lang_code == "hi" else "English"

    # 1) STT
    user_text = speech_to_text(audio_path, lang_code)

    # 2) LLM with short context
    assistant_text = generate_response(user_text, lang_code, lang_name, history)

    # 3) TTS
    audio_out = text_to_speech(assistant_text, lang_code)

    # 4) Persist if requested
    if persist_memory:
        # Update language preference + append messages
        user_mem["preferred_language"] = lang_code
        # Append two turns safely
        if user_text:
            history.append({"role": "user", "content": user_text})
        if assistant_text:
            history.append({"role": "assistant", "content": assistant_text})
        # Trim if oversized
        if len(history) > 4 * HISTORY_LIMIT:
            history = history[-(4 * HISTORY_LIMIT):]
        user_mem["history"] = history
        save_user_mem(user_id, user_mem)

    return assistant_text, audio_out

# ----------------------------
# Gradio UI
# ----------------------------
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=[
        gr.Audio(type="filepath", label="Speak"),
        gr.Radio(choices=["English", "Hindi"], value="English", label="Language"),
        gr.Textbox(label="User ID", value="guest", placeholder="e.g., phone/roll/email"),
        gr.Checkbox(label="Save memory for this user", value=True),
    ],
    outputs=[
        gr.Textbox(label="Response Text"),
        gr.Audio(label="Response Audio"),
    ],
    title="Real-Time Voice-to-Voice Chatbot (with Memory)",
    description="Select language, speak, and (optionally) save context by User ID for connected replies."
)

if __name__ == "__main__":
    iface.launch()
