import React from 'react';
import './App.css';
import Chatbot from './components/Chatbot';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Finaptive AI Chatbot</h1>
        <p>Multi-Source Query System</p>
      </header>
      <main className="App-main">
        <Chatbot />
      </main>
    </div>
  );
}

export default App;