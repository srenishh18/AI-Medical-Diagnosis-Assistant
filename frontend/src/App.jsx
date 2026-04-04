import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import Diagnose from './pages/Diagnose';
import Chat from './pages/Chat';
import './index.css';

const Navigation = () => {
  const location = useLocation();
  return (
    <nav className="navbar glass-panel slide-down">
      <Link to="/" className="nav-logo">⚕️ MedAI</Link>
      <div className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Home</Link>
        <Link to="/diagnose" className={`nav-link ${location.pathname === '/diagnose' ? 'active' : ''}`}>Diagnose</Link>
        <Link to="/chat" className={`nav-link ${location.pathname === '/chat' ? 'active' : ''}`}>Chat</Link>
      </div>
    </nav>
  );
};

function App() {
  return (
    <Router>
      <div className="app-container">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/diagnose" element={<Diagnose />} />
            <Route path="/chat" element={<Chat />} />
          </Routes>
        </main>

        <div className="background-shapes">
          <div className="shape shape-1"></div>
          <div className="shape shape-2"></div>
        </div>
      </div>
    </Router>
  );
}

export default App;
