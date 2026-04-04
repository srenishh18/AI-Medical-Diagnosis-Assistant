import React from 'react';
import { Link } from 'react-router-dom';
import '../index.css';

const Home = () => {
  return (
    <div className="page-container hero-page">
      <header className="hero text-center fade-in">
        <div className="logo-icon floating">⚕️</div>
        <h1 className="title-gradient">AI Medical Assistant</h1>
        <p className="subtitle text-light">Intelligent diagnostic guidance powered by AI and Machine Learning.</p>
        
        <div className="hero-buttons slide-up">
          <Link to="/diagnose" className="btn btn-primary btn-glow">
            Start Diagnosis
          </Link>
          <Link to="/chat" className="btn btn-outline">
            Chat with AI
          </Link>
        </div>
      </header>

      <section className="features-grid stagger-in">
        <div className="feature-card glass-panel">
          <div className="feature-icon">🎙️</div>
          <h3>Voice & Text Input</h3>
          <p>Describe your symptoms naturally using speech or text. We pinpoint the key indicators.</p>
        </div>
        <div className="feature-card glass-panel">
          <div className="feature-icon">🧠</div>
          <h3>ML Diagnostics</h3>
          <p>Our robust Random Forest model maps symptoms to the most probable conditions.</p>
        </div>
        <div className="feature-card glass-panel">
          <div className="feature-icon">💊</div>
          <h3>Immediate Precautions</h3>
          <p>Receive immediate, actionable safety steps before consulting a medical professional.</p>
        </div>
      </section>
    </div>
  );
};

export default Home;
