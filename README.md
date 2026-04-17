# AI Medical Voice Assistant 🩺🎙️

A powerful, high-performance web application designed for AI-driven clinical diagnostics. This assistant features real-time voice interaction, medical symptom extraction using Natural Language Processing (NLP), and diagnosis predictions powered by a custom-trained Random Forest model and the Gemini AI engine.

## ✨ Features

- **Clinical Diagnosis**: Uses a weighted Random Forest classifier to eliminate small-set bias for accurate disease prediction.
- **AI Voice Assistant**: Streaming AI voice responses directly from memory (no disk storage) for privacy and speed.
- **Dual Diagnosis Engine**:
  - **Local ML**: Fast, private matching against a supervised dataset for common conditions.
  - **AI Fallback**: Uses **Gemini 3.1 Flash Lite** for nuanced symptom analysis when local matching isn't sufficient.
- **Voice Interactions**: Speech-to-Text via OpenAI Whisper and Text-to-Speech via gTTS.

---

## 🚀 Getting Started

### 1. Requirements

Ensure you have the following installed:

- **Node.js** (v18+)
- **Python** (3.9+)
- **FFmpeg** (Required for audio processing)

### 2. Backend Setup

1. Open a terminal and navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file (already set up for you):
   - Add your `GEMINI_API_KEY` to `backend/.env`.

### 3. Frontend Setup

1. Open a new terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```

---

## ⚙️ Running the Application

### Start the Backend (FastAPI)

From the root directory:

```powershell
cd backend
..\.venv\Scripts\uvicorn.exe main:app --reload
```

_The backend will run at [http://localhost:8000](http://localhost:8000)_

### Start the Frontend (Vite + React)

From the root directory in a **new** terminal:

```powershell
cd frontend
npm run dev
```

_The frontend will run at [http://localhost:5173](http://localhost:5173)_

---

## 🛠️ Tech Stack

- **Frontend**: React, Vite, Vanilla CSS (Premium Aethetics)
- **Backend**: FastAPI, Uvicorn
- **AI/ML**: Google Gemini API (3.1 Flash Lite), scikit-learn (Random Forest), Whisper STT, gTTS
- **NLP**: SpaCy

## 📝 Disclaimer

This application is an AI-powered diagnostic tool for educational and informational purposes. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare professional for serious medical issues.


