import React, { useState, useRef, useEffect } from 'react';
import '../index.css';

const Diagnose = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState('');
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const aiAudioPlayerRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ audio: true }).catch(err => {
      console.error("Microphone access denied:", err);
    });
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioURL(URL.createObjectURL(audioBlob));
        await sendAudioToBackend(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setResult(null);
    } catch (error) {
      alert("Microphone permission error.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    setIsProcessing(true);
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");

    try {
      const response = await fetch("http://localhost:8000/api/diagnose/audio", {
        method: "POST",
        body: formData,
      });
      handleResponse(response);
    } catch (error) {
      alert("Failed to reach server.");
      setIsProcessing(false);
    }
  };

  const sendTextToBackend = async () => {
    if (!inputText.trim()) return;
    setIsProcessing(true);
    setResult(null);
    setAudioURL('');

    try {
      const response = await fetch("http://localhost:8000/api/diagnose/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });
      handleResponse(response);
    } catch (error) {
      alert("Failed to reach server.");
      setIsProcessing(false);
    }
  };

  const handleResponse = async (response) => {
    try {
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      
      setResult(data);
      if (aiAudioPlayerRef.current && data.audio_url) {
        aiAudioPlayerRef.current.src = `http://localhost:8000${data.audio_url}`;
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
        <p>Speak or type your symptoms to get ML-powered insights.</p>
      </header>

      <div className="input-cards-container stagger-in">
        <div className="input-card glass-panel">
          <h3>Voice Input</h3>
          <div className={`mic-container ${isRecording ? 'pulse' : ''}`}>
            <button 
              className={`mic-button ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
            >
              {isRecording ? '⏹' : '🎤'}
            </button>
          </div>
          <p className="status-text">{isRecording ? "Listening..." : "Tap to speak"}</p>
        </div>

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
                <audio ref={aiAudioPlayerRef} controls src={`http://localhost:8000${result.audio_url}`} className="styled-audio"></audio>
              </div>
            )}
            
          </div>
        </section>
      )}
    </div>
  );
};

export default Diagnose;
