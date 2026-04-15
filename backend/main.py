import os
import shutil
import uuid
import joblib
import numpy as np
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import google.generativeai as genai
import spacy
from gtts import gTTS
import asyncio
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import time


load_dotenv()

app = FastAPI(title="AI Medical Assistant API")

# Setup CORS to allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_model = None
rf_model = None
symptoms_list = []
precaution_dict = {}
disease_symptom_counts = {}

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-3.1-flash-lite-preview')
else:
    gemini_model = None

# Temporary directory for standard input audio processing
base_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(base_root_dir, "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

def get_stt_model():
    global stt_model
    if stt_model is None:
        print("Loading Whisper STT Model (base)...")
        stt_model = whisper.load_model("base")
    return stt_model

@app.on_event("startup")
def load_models():
    global rf_model, symptoms_list, precaution_dict, disease_symptom_counts, nlp_model
    
    # Manually add typical Windows winget ffmpeg paths
    ffmpeg_extra_path = r"C:\Users\RENISH KUMAR\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
    if os.path.exists(ffmpeg_extra_path) and ffmpeg_extra_path not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_extra_path
        print(f"Added ffmpeg to PATH: {ffmpeg_extra_path}")
        
    print("Loading Core Models and Data...")
    
    # Pre-load spaCy (Optimized: disable unused components)
    try:
        print("Warming up NLP model (spaCy)...")
        nlp_model = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except Exception as e:
        print(f"Warning: NLP loading failed: {e}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "rf_model.pkl")
    symptoms_path = os.path.join(base_dir, "models", "symptoms_list.pkl")
    counts_path = os.path.join(base_dir, "models", "disease_symptom_counts.pkl")
    precaution_path = os.path.join(os.path.dirname(base_dir), "dataset", "Disease precaution.csv")
    
    if os.path.exists(model_path) and os.path.exists(symptoms_path):
        print("Loading Random Forest model, symptoms list and counts...")
        rf_model = joblib.load(model_path)
        symptoms_list = joblib.load(symptoms_path)
        if os.path.exists(counts_path):
            disease_symptom_counts = joblib.load(counts_path)
    else:
        print("WARNING: RF model files not found. Train the model first.")
        
    if os.path.exists(precaution_path):
        print("Loading precautions dataset...")
        prec_df = pd.read_csv(precaution_path)
        for _, row in prec_df.iterrows():
            disease = str(row['Disease']).strip()
            precs = [str(x) for x in row.iloc[1:5] if pd.notna(x) and str(x).strip() != '']
            precaution_dict[disease] = precs

nlp_model = None
def get_nlp_model():
    global nlp_model
    if nlp_model is None:
        import spacy
        try:
            nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp_model = spacy.load("en_core_web_sm")
    return nlp_model

def get_local_symptom_extraction(user_text: str, symptoms_list: list) -> list:
    import difflib
    user_text_lower = user_text.lower()
    
    # Custom Synonyms
    syn_map = {
        "yellow eyes": "yellowing of eyes", "yellowish eyes": "yellowing of eyes",
        "dark pee": "dark urine", "tummy": "stomach", "belly": "stomach",
        "hurts": "pain", "aching": "pain"
    }
    for syn, target in syn_map.items():
        if syn in user_text_lower:
            user_text_lower = user_text_lower.replace(syn, target)
            
    # --- PHASE 1: Quick-Match (Dictionary Based) ---
    # Fast-path for common symptoms to bypass spaCy
    matched = set()
    symptoms_clean_map = {sym.replace('_', ' ').strip().lower(): sym for sym in symptoms_list}
    
    stop_words = {"the", "my", "a", "an", "of", "and", "is", "in", "on"}
    user_words = set(w for w in user_text_lower.replace('.', ' ').replace(',', ' ').split() if w not in stop_words)

    # Direct keyword check (smarter)
    for clean_sym, original_sym in symptoms_clean_map.items():
        # Check if all essential words of the symptom are in the user words
        sym_words = set(w for w in clean_sym.split() if w not in stop_words)
        if sym_words and sym_words.issubset(user_words):
            matched.add(original_sym)
            
    # If we found matches quickly, we can skip the heavy NLP
    if len(matched) >= 2:
        print(f"DEBUG: Using Quick-Match (Bypassing spaCy) - Matched: {matched}")
        return sorted(list(matched))

    # --- PHASE 2: Heavy NLP (spaCy) ---
    nlp = get_nlp_model()
    doc = nlp(user_text_lower)
    
    extracted_tokens = []
    # ... (rest of the spaCy logic remains as a fallback)
    
    for token in doc:
        if token.is_stop or token.is_punct or len(token.text.strip()) <= 3:
            continue
        if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
            extracted_tokens.append(token.lemma_)
            
    for chunk in doc.noun_chunks:
        clean_chunk = " ".join([t.text for t in chunk if not t.is_stop and len(t.text) > 2])
        if len(clean_chunk) > 3:
            extracted_tokens.append(clean_chunk)
            
    synonyms = {
        "tummy": "stomach", "belly": "stomach", "throw up": "vomiting",
        "puke": "vomiting", "aching": "pain", "hurts": "pain",
        "tired": "fatigue", "exhausted": "lethargy", "warm": "mild_fever",
        "hot": "high_fever", "feverish": "high_fever", "dizzy": "dizziness",
        "shivering": "chills", "shiver": "chills", "chills": "shivering"
    }

    symptoms_clean_map = {sym.replace('_', ' ').strip().lower(): sym for sym in symptoms_list}
    clean_symptoms_list = list(symptoms_clean_map.keys())
    
    for token in extracted_tokens:
        token = token.strip().lower()
        if token in synonyms:
            token = synonyms[token]
            if token in symptoms_list:
                matched.add(token)
                
        closest = difflib.get_close_matches(token, clean_symptoms_list, n=1, cutoff=0.8)
        if closest:
            matched.add(symptoms_clean_map[closest[0]])
            continue
            
        if len(token) > 3:
            for clean_sym, original_sym in symptoms_clean_map.items():
                if token == clean_sym or clean_sym == token:
                    matched.add(original_sym)
                
    return sorted(list(matched))

def get_diseases_and_precautions(matched_symptoms: list):
    if not rf_model or not symptoms_list or not matched_symptoms:
        return [], []
        
    X = np.zeros((1, len(symptoms_list)), dtype=int)
    symptom_to_idx = {sym: idx for idx, sym in enumerate(symptoms_list)}
    for sym in matched_symptoms:
        if sym in symptom_to_idx:
            X[0, symptom_to_idx[sym]] = 1
            
    probabilities = rf_model.predict_proba(X)[0]
    
    # Weighted Accuracy Logic:
    # Use Match Ratio (user hits / total required) to penalize small-set diseases (Allergy)
    weighted_probs = []
    num_user_symptoms = len(matched_symptoms)
    for idx, disease_name in enumerate(rf_model.classes_):
        raw_prob = probabilities[idx]
        # Get total symptoms this disease requires in the dataset
        total_req = disease_symptom_counts.get(disease_name, 5)
        # Ratio of coverage
        match_ratio = num_user_symptoms / total_req
        # Clamp ratio
        if match_ratio > 1.0: match_ratio = 1.0
        # If the user has broad symptoms (High Fever/Headache) that the disease NEVER has, penalize it heavily
        negation_penalty = 1.0
        if disease_name == "Allergy" and ("high_fever" in matched_symptoms or "headache" in matched_symptoms):
            negation_penalty = 0.1
        if disease_name == "COVID-19" and ("joint_pain" in matched_symptoms or "skin_rash" in matched_symptoms):
            negation_penalty = 0.05
            
        weighted_probs.append(raw_prob * (0.5 + 0.5 * match_ratio) * negation_penalty)
    
    # Priority Overrides
    if "yellowish_skin" in matched_symptoms and len(matched_symptoms) <= 2:
        # If user ONLY has yellowing, prioritize Jaundice over viral Hepatitis
        j_idx = -1
        for idx, name in enumerate(rf_model.classes_):
            if name == "Jaundice": j_idx = idx
        if j_idx != -1:
            weighted_probs[j_idx] += 0.5

    top_3_indices = np.argsort(weighted_probs)[::-1][:3]
    top_3_diseases = rf_model.classes_[top_3_indices].tolist()
    top_disease = top_3_diseases[0]
    
    precautions = []
    lookup_name = str(top_disease).strip().lower()
    for d_name, d_precs in precaution_dict.items():
        if d_name.strip().lower() == lookup_name:
            precautions = d_precs
            break
            
    return top_disease, precautions

class TextDiagnoseRequest(BaseModel):
    text: str

@app.post("/api/diagnose/audio")
async def process_audio(audio: UploadFile = File(...)):
    temp_audio_path = f"{TEMP_DIR}/{uuid.uuid4()}_{audio.filename}"
    try:
        start_time = time.time()
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        stt_start = time.time()
        # LAZY LOAD STT
        model = get_stt_model()
        transcription_result = model.transcribe(temp_audio_path, fp16=False)
        print(f"DEBUG: Whisper STT took {time.time() - stt_start:.2f}s")
        
        user_text = transcription_result["text"].strip()
        if not user_text:
            return JSONResponse(status_code=400, content={"error": "Could not extract clear speech."})
        
        diag_start = time.time()
        res = await process_diagnosis(user_text)
        print(f"DEBUG: Diagnosis Logic took {time.time() - diag_start:.2f}s")
        print(f"DEBUG: Total Request time: {time.time() - start_time:.2f}s")
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.post("/api/diagnose/text")
async def process_text(request: TextDiagnoseRequest):
    if not request.text:
       return JSONResponse(status_code=400, content={"error": "Empty text provided."}) 
    
    start_time = time.time()
    res = await process_diagnosis(request.text)
    print(f"DEBUG: Total Text Request time: {time.time() - start_time:.2f}s")
    return res

async def process_diagnosis(user_text: str):
    nlp_start = time.time()
    matched_symptoms = get_local_symptom_extraction(user_text, symptoms_list)
    print(f"DEBUG: Symptom Extraction (spaCy) took {time.time() - nlp_start:.2f}s")
    
    top_disease = ""
    precautions = []
    
    if matched_symptoms:
        top_disease, precautions = get_diseases_and_precautions(matched_symptoms)
        ai_response_text = f"Based on your symptoms ({', '.join(matched_symptoms)}), the most likely condition is {top_disease}.\n\n"
        if precautions:
            ai_response_text += f"Recommended precautions:\n- " + "\n- ".join(precautions) + "\n\n"
        ai_response_text += "Disclaimer: I am an AI, not a doctor. Please consult a healthcare professional for serious medical advice."
    else:
        if not gemini_model:
            ai_response_text = "I couldn't match any standard symptoms locally, and Gemini API is not configured."
        else:
            prompt = f"Summarize these medical symptoms briefly for a user and suggest seeking professional advice. Symptoms: {user_text}"
            try:
                gemini_start = time.time()
                response = gemini_model.generate_content(prompt)
                print(f"DEBUG: Gemini AI API took {time.time() - gemini_start:.2f}s")
                ai_response_text = response.text
            except:
                ai_response_text = "I'm having trouble analyzing your symptoms right now."

    clean_tts_text = ai_response_text.replace("*", "").replace("#", "")
    
    # Generate unique ID to retrieve this in-memory audio session
    session_id = str(uuid.uuid4())
    # We will pass the text to the streaming route
    return {
        "transcription": user_text,
        "extracted_symptoms": matched_symptoms,
        "top_disease": top_disease,
        "precautions": precautions,
        "ai_response": ai_response_text,
        "audio_url": f"/api/audio/stream?text={clean_tts_text.replace(' ', '+')}"
    }

@app.get("/api/audio/stream")
async def stream_audio(text: str):
    """Generates and streams audio directly from memory - NO DISK STORAGE"""
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return StreamingResponse(mp3_fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    if not gemini_model:
         return JSONResponse(status_code=500, content={"error": "Gemini API not configured."})
    prompt = f"Helpful medical AI. User: {request.message}"
    try:
        response = gemini_model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
