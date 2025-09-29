# App.py - Streamlit Speech Analyzer

import os
import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS

# --- Setup folders ---
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Load Whisper model ---
@st.cache_resource(show_spinner=True)
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.eval()
    return processor, model

processor, model = load_whisper_model()

# --- Helper functions ---
def convert_to_wav(file_path):
    """Convert MP3 or video files to WAV for librosa compatibility"""
    if not file_path.lower().endswith(".wav"):
        wav_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path).rsplit(".", 1)[0] + ".wav")
        if file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            clip = VideoFileClip(file_path)
            clip.audio.write_audiofile(wav_path)
        else:
            audio = AudioSegment.from_file(file_path)
            audio.export(wav_path, format="wav")
        return wav_path
    return file_path

def transcribe_audio(audio_file_path):
    wav_path = convert_to_wav(audio_file_path)
    audio_input, _ = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=1024)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

def convert_to_fluent_audio(text, file_path):
    mp3_file_path = os.path.join(
        OUTPUT_FOLDER, os.path.basename(file_path).rsplit(".", 1)[0] + "_fluent.mp3"
    )
    tts = gTTS(text=text, lang="en")
    tts.save(mp3_file_path)
    return mp3_file_path

# --- Streamlit UI ---
st.title("🎤 Speech Analyzer")
st.write("Upload an audio or video file, and get its transcription along with fluent audio output.")

uploaded_file = st.file_uploader(
    "Upload Audio/Video File",
    type=["wav", "mp3", "mp4", "mkv", "avi", "mov"]
)

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("🔊 Processing your file. This may take a few moments...")

    # Transcribe & convert to fluent audio
    transcription = transcribe_audio(file_path)
    fluent_audio_path = convert_to_fluent_audio(transcription, file_path)

    # Display results
    st.subheader("📄 Transcription Result")
    st.write(transcription)

    st.subheader("🎧 Fluent Audio")
    st.audio(fluent_audio_path, format="audio/mp3")

    # Download button
    st.download_button(
        label="⬇️ Download Fluent Audio",
        data=open(fluent_audio_path, "rb").read(),
        file_name=os.path.basename(fluent_audio_path),
        mime="audio/mp3"
    )

    # Optional: clean up uploaded file
    # os.remove(file_path)
