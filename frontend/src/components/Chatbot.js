import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './Chatbot.css';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSystemLoading, setIsSystemLoading] = useState(true);
  const [systemStatus, setSystemStatus] = useState(null);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchSystemStatus = React.useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      const data = response.data;
      setSystemStatus(data);
      
      // Keep loading screen if system is still loading
      if (data.status === 'loading') {
        setIsSystemLoading(true);
        // Poll again after 500ms for more responsive updates
        setTimeout(fetchSystemStatus, 500);
      } else {
        setIsSystemLoading(false);
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      setIsSystemLoading(false);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchSystemStatus();
    // Add welcome message
    setMessages([
      {
        id: 1,
        type: 'bot',
        content: 'Hello! I\'m your Finaptive AI Assistant. I can help you query SQL databases, Excel files, CSV files, and emails. What would you like to know?',
        timestamp: new Date()
      }
    ]);
  }, [fetchSystemStatus]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    // Add retrieving message immediately
    const retrievingMessage = {
      id: Date.now() + 1,
      type: 'bot',
      content: 'Retrieving and Analyzing...',
      timestamp: new Date(),
      isRetrieving: true
    };

    const startTime = Date.now(); // Start timing
    
    setMessages(prev => [...prev, userMessage, retrievingMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: inputMessage,
        session_id: 'web-session-' + Date.now()
      });

      const endTime = Date.now(); // End timing
      const processingTime = ((endTime - startTime) / 1000).toFixed(2); // Convert to seconds

      // Replace retrieving message with actual response
      const botMessage = {
        id: retrievingMessage.id, // Keep same ID to replace
        type: 'bot',
        content: response.data.response,
        timestamp: new Date(),
        sources: response.data.sources,
        processingTime: processingTime,
        isRetrieving: false
      };
      
      setMessages(prev => prev.map(msg => 
        msg.id === retrievingMessage.id ? botMessage : msg
      ));
    } catch (error) {
      const endTime = Date.now(); // End timing even for errors
      const processingTime = ((endTime - startTime) / 1000).toFixed(2);

      // Replace retrieving message with error
      const errorMessage = {
        id: retrievingMessage.id, // Keep same ID to replace
        type: 'bot',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        processingTime: processingTime,
        isError: true,
        isRetrieving: false
      };
      
      setMessages(prev => prev.map(msg => 
        msg.id === retrievingMessage.id ? errorMessage : msg
      ));
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(e);
    }
  };

  const formatMessage = (content) => {
    // Simple formatting for line breaks
    return content.split('\n').map((line, index) => (
      <React.Fragment key={index}>
        {line}
        {index < content.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <div className="chatbot-container">
      <div className="chat-header">
        <div className="status-indicator">
          <span className={`status-dot ${systemStatus?.status === 'active' ? 'active' : 'inactive'}`}></span>
          <span className="status-text">
            {systemStatus?.status === 'active' ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="available-sources">
          Sources: {systemStatus?.available_sources?.join(', ') || 'Loading...'}
        </div>
      </div>

      {isSystemLoading && (
        <div className="system-loading-overlay">
          <div className="loading-content">
            <div className="loading-spinner"></div>
            <h3>Loading Finaptive AI System</h3>
            <p>{systemStatus?.system_info || 'Initializing RAG discovery and preloading data sources...'}</p>
            <div className="loading-steps">
              <div className="loading-step">📊 Building vector index</div>
              <div className="loading-step">📈 Loading Excel data (VW_PBI)</div>
              <div className="loading-step">📄 Loading CSV files</div>
              <div className="loading-step">🤖 Preparing AI agents</div>
            </div>
          </div>
        </div>
      )}

      <div className="messages-container">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.type}`}>
            <div className="message-content">
              <div className={`message-bubble ${message.isError ? 'error' : ''} ${message.isRetrieving ? 'retrieving' : ''}`}>
                {message.isRetrieving ? (
                  <div className="retrieving-content">
                    <div className="retrieving-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    {formatMessage(message.content)}
                  </div>
                ) : (
                  formatMessage(message.content)
                )}
              </div>
              <div className="message-time">
                {message.timestamp.toLocaleTimeString()}
                {message.processingTime && message.type === 'bot' && (
                  <span className="processing-time">
                    • {message.processingTime}s
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-form" onSubmit={sendMessage}>
        <div className="input-container">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me about your data..."
            className="message-input"
            rows="1"
            disabled={isLoading}
          />
          <button 
            type="submit" 
            className="send-button"
            disabled={isLoading || !inputMessage.trim()}
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default Chatbot;