import React, { useState, useEffect, useRef } from 'react';
import { sendMessageToContent } from '../utils/messaging';
import type { Message, AgentStatus } from '../types/messages';

const SidePanel: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [status, setStatus] = useState<AgentStatus>('idle');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load messages from storage on mount
  useEffect(() => {
    chrome.storage.local.get(['chatHistory'], (result) => {
      if (result.chatHistory) {
        setMessages(result.chatHistory);
      }
    });
  }, []);

  // Save messages to storage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      chrome.storage.local.set({ chatHistory: messages });
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setStatus('thinking');

    try {
      // Send message to content script
      const response = await sendMessageToContent({
        type: 'USER_PROMPT',
        payload: { prompt: inputValue },
      });

      setStatus('acting');

      // Simulate agent response (replace with actual LLM call later)
      setTimeout(() => {
        const agentMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'agent',
          content: response?.message || 'I received your request and inspected the DOM.',
          timestamp: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, agentMessage]);
        setStatus('idle');
      }, 1000);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'agent',
        content: 'Error: Could not communicate with the page. Make sure you have a tab open.',
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, errorMessage]);
      setStatus('idle');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearHistory = () => {
    setMessages([]);
    chrome.storage.local.remove(['chatHistory']);
  };

  const getStatusColor = () => {
    switch (status) {
      case 'thinking':
        return 'bg-yellow-500';
      case 'acting':
        return 'bg-blue-500';
      default:
        return 'bg-green-500';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'thinking':
        return 'Thinking...';
      case 'acting':
        return 'Acting...';
      default:
        return 'Idle';
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 shadow-sm">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-800">Big Brother</h1>
          <button
            onClick={clearHistory}
            className="text-sm text-gray-600 hover:text-gray-800 px-3 py-1 rounded hover:bg-gray-100 transition-colors"
          >
            Clear History
          </button>
        </div>
        
        {/* Status Indicator */}
        <div className="flex items-center gap-2 mt-2">
          <div className={`w-2 h-2 rounded-full ${getStatusColor()} animate-pulse`} />
          <span className="text-sm text-gray-600">{getStatusText()}</span>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <p className="text-lg mb-2">No messages yet</p>
              <p className="text-sm">Start a conversation to interact with the page</p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-800 border border-gray-200'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                <p
                  className={`text-xs mt-1 ${
                    message.role === 'user' ? 'text-blue-200' : 'text-gray-400'
                  }`}
                >
                  {new Date(message.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white px-4 py-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter your prompt (e.g., 'Change my username')"
            disabled={status !== 'idle'}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || status !== 'idle'}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
          >
            Send
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Press Enter to send • The agent will inspect and interact with the current page
        </p>
      </div>
    </div>
  );
};

export default SidePanel;
