// Message types for communication between components

export type AgentStatus = 'idle' | 'thinking' | 'acting';

export interface Message {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: string;
}

export interface ChromeMessage {
  type: string;
  payload?: any;
  target?: 'content' | 'background' | 'sidepanel';
}

export interface UserPromptMessage extends ChromeMessage {
  type: 'USER_PROMPT';
  payload: {
    prompt: string;
  };
  target: 'content';
}

export interface ContentResponse {
  success: boolean;
  message?: string;
  domSnapshot?: string;
  error?: string;
}
