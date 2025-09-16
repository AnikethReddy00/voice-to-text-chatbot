
import os
from tempfile import NamedTemporaryFile

import gradio as gr
import whisper
from gtts import gTTS
from groq import Groq

# ----------------------------
# Groq Client (kept as-is)
# ----------------------------
client = Groq(
    api_key=""  # keep your key here if you prefer
)

# Load Whisper once
asr_model = whisper.load_model("base")  # use "tiny"/"small" for faster startup if needed

# Map UI choice -> ISO code + readable name
LANG_MAP = {
    "English": ("en", "English"),
    "Hindi": ("hi", "Hindi"),
}

def speech_to_text(audio_path: str, lang_code: str) -> str:
    """Transcribe using Whisper, forcing the selected language (en/hi)."""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        out = asr_model.transcribe(audio_path, language=lang_code)
        return (out.get("text") or "").strip()
    except Exception as e:
        return f"[ASR error] {e}"

def generate_response(text: str, lang_code: str, lang_name: str) -> str:
    """Ask Groq LLM to reply strictly in the selected language."""
    text = (text or "").strip()
    if not text:
        return "I didn't catch that. Please try again." if lang_code == "en" else "मैं समझ नहीं पाया। कृपया फिर से बोलें।"
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # current, supported Groq model
            messages=[
                {
                    "role": "system",
                    "content": f"Reply ONLY in {lang_name}. If the user speaks another language, translate their message and respond in {lang_name}."
                },
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[Groq error] {e}" if lang_code == "en" else f"[Groq त्रुटि] {e}"

def text_to_speech(text: str, lang_code: str) -> str:
    """Convert model reply to speech in the selected language."""
    try:
        tts = gTTS(text or ("Sorry, I couldn't generate a response." if lang_code == "en" else "क्षमा करें, मैं उत्तर नहीं दे सका।"),
                   lang=lang_code)
        f = NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(f.name)
        return f.name
    except Exception as e:
        # If TTS fails, return None audio but keep text
        return None

def chatbot_pipeline(audio_path, language_choice):
    """
    language_choice: "English" or "Hindi" (UI dropdown)
    """
    lang_code, lang_name = LANG_MAP.get(language_choice, ("en", "English"))
    try:
        # 1) STT in selected language
        text_input = speech_to_text(audio_path, lang_code)

        # 2) LLM reply in selected language
        response_text = generate_response(text_input, lang_code, lang_name)

        # 3) TTS in selected language
        response_audio_path = text_to_speech(response_text, lang_code)

        return response_text, response_audio_path
    except Exception as e:
        return (f"[Pipeline error] {e}" if lang_code == "en" else f"[पाइपलाइन त्रुटि] {e}"), None

# ----------------------------
# Gradio UI
# ----------------------------
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=[
        gr.Audio(type="filepath", label="Speak"),
        gr.Radio(choices=["English", "Hindi"], value="English", label="Language"),
    ],
    outputs=[
        gr.Textbox(label="Response Text"),
        gr.Audio(label="Response Audio")
    ],
    title="Real-Time Voice-to-Voice Chatbot"
)

# Shareable public URL
if __name__ == "__main__":
    iface.launch()


