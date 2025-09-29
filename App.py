# App.py - Streamlit Speech Analyzer

import os
import streamlit as st
from gtts import gTTS
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from moviepy.editor import VideoFileClip
import imageio_ffmpeg  # ensures ffmpeg works with moviepy

# -------------------- Setup --------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------- Load Whisper --------------------
@st.cache_resource(show_spinner=True)
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.eval()
    return processor, model

processor, model = load_whisper_model()

# -------------------- Functions --------------------
def transcribe_audio(audio_file_path):
    audio_input, _ = librosa.load(audio_file_path, sr=16000, mono=True)
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_length=1024)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

def convert_to_fluent_audio(text, wav_file_path):
    tts = gTTS(text=text, lang="en")
    mp3_file_path = os.path.join(
        OUTPUT_FOLDER, os.path.basename(wav_file_path).rsplit(".", 1)[0] + "_fluent.mp3"
    )
    tts.save(mp3_file_path)
    return mp3_file_path

# -------------------- Streamlit UI --------------------
st.title("🎤 Speech Analyzer")
st.write("Upload an audio or video file to get transcription and fluent speech output.")

uploaded_file = st.file_uploader(
    "Upload Audio/Video File",
    type=["wav", "mp3", "mp4", "mkv", "avi", "mov"]
)

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert video to audio if needed
    if uploaded_file.name.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
        audio_path = os.path.join(
            UPLOAD_FOLDER, uploaded_file.name.rsplit(".", 1)[0] + ".wav"
        )
        clip = VideoFileClip(file_path)
        clip.audio.write_audiofile(audio_path)
        file_path = audio_path

    # Transcribe & convert to fluent audio
    with st.spinner("🔊 Processing..."):
        transcription = transcribe_audio(file_path)
        fluent_audio_path = convert_to_fluent_audio(transcription, file_path)

    # Show transcription
    st.subheader("📄 Transcription Result")
    st.write(transcription)

    # Play fluent audio
    st.subheader("🎧 Fluent Audio")
    st.audio(fluent_audio_path, format="audio/mp3")

    # Download button
    st.download_button(
        label="⬇️ Download Fluent Audio",
        data=open(fluent_audio_path, "rb").read(),
        file_name=os.path.basename(fluent_audio_path),
        mime="audio/mp3"
    )
