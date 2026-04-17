import React, { useState, useRef } from 'react';
import '../index.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const fetchWithRetry = async (url, options, retries = 3) => {
  let lastError = null;

  for (let attempt = 1; attempt <= retries; attempt += 1) {
    try {
      return await fetch(url, options);
    } catch (error) {
      lastError = error;
      if (attempt < retries) {
        await sleep(500 * attempt);
      }
    }
  }

  throw lastError;
};

const Diagnose = () => {
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);

  const aiAudioPlayerRef = useRef(null);

  const sendTextToBackend = async () => {
    if (!inputText.trim()) return;
    setIsProcessing(true);
    setResult(null);

    try {
      const response = await fetchWithRetry(`${API_BASE_URL}/api/diagnose/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });
      await handleResponse(response);
    } catch (error) {
      alert("Failed to reach server. Make sure the backend is running on port 8000 and try again.");
      setIsProcessing(false);
    }
  };

  const handleResponse = async (response) => {
    try {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || data.detail || `Request failed (${response.status})`);
      }
      if (data.error) throw new Error(data.error);
      
      setResult(data);
      if (data.transcription) {
        setInputText(data.transcription);
      }
      if (aiAudioPlayerRef.current && data.audio_url) {
        aiAudioPlayerRef.current.src = `${API_BASE_URL}${data.audio_url}`;
        aiAudioPlayerRef.current.play().catch(e => console.warn(e));
      }
    } catch (e) {
      alert("Error: " + e.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="page-container diagnose-page">
      <header className="page-header text-center slide-down">
        <h2>Symptom Diagnosis</h2>
        <p>Type your symptoms to get ML-powered insights.</p>
      </header>

      <div className="input-cards-container stagger-in">
        <div className="input-card glass-panel">
          <h3>Text Input</h3>
          <textarea 
            className="text-input" 
            placeholder="E.g., I have a severe headache and I'm feeling nauseous..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isProcessing}
          />
          <button 
            className="btn btn-primary" 
            onClick={sendTextToBackend}
            disabled={isProcessing || !inputText.trim()}
          >
            {isProcessing ? "Analyzing..." : "Analyze Symptoms"}
          </button>
        </div>
      </div>

      {result && (
        <section className="results-section slide-up">
          <div className="card results-card glass-panel">
            <h3 className="section-title">Analysis Results</h3>
            <p><strong>Input:</strong> "{result.transcription}"</p>
            
            <div className="tags-container">
              <strong>Extracted Symptoms: </strong>
              {result.extracted_symptoms && result.extracted_symptoms.length > 0 ? 
                result.extracted_symptoms.map((sym, i) => <span key={i} className="tag tag-blue">{sym}</span>) :
                <span className="text-dim">No strict matches found. Used AI fallback.</span>
              }
            </div>

            {result.top_disease && (
              <div className="ml-predictions">
                <h4 className="gradient-text">Probable Diagnosis</h4>
                <div className="top-disease-highlight">
                  <span className="disease-name-main">{result.top_disease}</span>
                  <span className="fire-icon">🔥</span>
                </div>
              </div>
            )}

            {result.precautions && result.precautions.length > 0 && (
              <div className="precautions-box">
                <h4>Safety Precautions</h4>
                <ul>
                  {result.precautions.map((p, i) => <li key={i}>{p}</li>)}
                </ul>
              </div>
            )}

            <div className="ai-summary">
              <h4>AI Overview</h4>
              <p>{result.ai_response}</p>
            </div>
            
            {result.audio_url && (
              <div className="audio-player-wrapper">
                <p>Listen to Diagnosis</p>
                <audio ref={aiAudioPlayerRef} controls src={`${API_BASE_URL}${result.audio_url}`} className="styled-audio"></audio>
              </div>
            )}
            
          </div>
        </section>
      )}
    </div>
  );
};

export default Diagnose;
