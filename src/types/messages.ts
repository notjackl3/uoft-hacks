// Message types for communication between components

export type AgentStatus = 'idle' | 'thinking' | 'acting';

export interface Message {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: string;
  // Optional metadata for step messages
  stepInfo?: {
    stepNumber: number;
    totalSteps: number;
    action: string;
    targetIndex: number | null;
    confidence: number;
  };
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

// Message to get page features without performing any action
export interface GetFeaturesMessage extends ChromeMessage {
  type: 'GET_FEATURES';
  target: 'content';
}

// Message to execute an action on a specific element
export interface ExecuteActionMessage extends ChromeMessage {
  type: 'EXECUTE_ACTION';
  payload: {
    action: 'CLICK' | 'TYPE' | 'SCROLL' | 'WAIT';
    targetIndex: number | null;
    textInput?: string;
  };
  target: 'content';
}

// Message to highlight an element (without executing)
export interface HighlightElementMessage extends ChromeMessage {
  type: 'HIGHLIGHT_ELEMENT';
  payload: {
    targetIndex: number;
    duration?: number; // ms, default 3000
  };
  target: 'content';
}

// Message to clear all highlights
export interface ClearHighlightsMessage extends ChromeMessage {
  type: 'CLEAR_HIGHLIGHTS';
  target: 'content';
}

// Message to wait for a user interaction on the page (guidance-only mode)
export interface WaitForEventMessage extends ChromeMessage {
  type: 'WAIT_FOR_EVENT';
  payload: {
    event: 'click' | 'input' | 'scroll';
    targetIndex?: number | null; // required for click/input; ignored for scroll
    timeoutMs?: number;
  };
  target: 'content';
}

export interface PageFeature {
  index: number;
  type: 'input' | 'button' | 'link';
  text: string;
  selector: string;
  href?: string;
  placeholder?: string;
  aria_label?: string;
}

export interface ContentResponse {
  success: boolean;
  message?: string;
  error?: string;
  // Page info
  pageTitle?: string;
  pageUrl?: string;
  // Features extracted from page
  features?: PageFeature[];
}
