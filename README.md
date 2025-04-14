# 🧠 Fluent Speech Correction Web App

## 🚩 Problem Statement

**People who stammer face communication challenges because traditional speech recognition systems struggle to process disfluencies like stutters and long pauses.**  
These systems often misinterpret or skip words, leading to inaccurate transcriptions and failed emotion detection — decreasing accessibility for individuals with speech impairments.

There is a clear need for an **AI-powered speech processing system** that can:
- Detect **stammering patterns**
- Analyze **emotions**
- Convert dysfluent speech to **clear, fluent output**
This app aims to bridge that gap and empower inclusive communication.

---

## 📌 Project Overview

The **Fluent Speech Correction App** is an AI-driven Flask web application that allows users to upload audio/video files and returns:
- ✅ Accurate **transcription** using OpenAI Whisper
- 😊 **Emotion detection** from transcribed text
- 🧠 **Stammer severity rating**
- 🔊 A **fluent audio output** generated from the clean text using Google TTS

It’s built for accessibility, aiming to support people with speech disfluencies in achieving smooth and effective communication.

---

## 🛠️ Implementation Details

### 🔧 Backend Stack:
- **Flask**: Powers the web interface and routes.
- **OpenAI Whisper**: For robust, multilingual speech-to-text transcription.
- **Google TTS (gTTS)**: Converts fluent text back into natural-sounding speech.
- **Librosa**: Audio file loading and preprocessing.
- **MoviePy**: Extracts audio from video files if needed.
- **Pyngrok**: For creating public URLs (especially for Google Colab users).

### 📂 Flow:
1. User uploads an audio or video file
2. Audio is extracted (if video) and preprocessed
3. Whisper transcribes the speech
4. Emotion and stammer detection run on transcript/audio
5. gTTS generates fluent audio from the cleaned transcript
6. All results are shown in the browser with download options

---

## 🚀 How to Install & Run

### 🔧 Prerequisites:
- Python ≥ 3.7
- pip
- Virtual environment (recommended)

### 🧪 Installation:
```bash
git clone https://github.com/Chandrashekar0123/fluent-speech-app.git
cd fluent-speech-app
pip install -r requirements.txt

markdown
Copy
Edit
## ▶️ Run Locally

```bash
python app.py
🧑‍💻 Or Run on Google Colab
Just paste the code in a Colab cell.
pyngrok will generate a public URL to access the app.

📷 Features Demo (Sample UI)
✅ Upload audio/video

🧠 Get emotion & stammer severity

📃 Read transcription

🔊 Listen & download fluent audio

📈 Future Scope
Real stammer and emotion detection using deep learning

Speaker diarization & multi-language support

Feedback-based audio enhancement

🧾 License
This project is released under the MIT License — feel free to use, share, and build upon it.
