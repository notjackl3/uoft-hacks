// Backend API client for the tutor agent

const API_BASE_URL = 'http://localhost:8000';

export interface PageFeature {
  index: number;
  type: 'input' | 'button' | 'link';
  text: string;
  selector: string;
  href?: string;
  placeholder?: string;
  aria_label?: string;
  value_len?: number;
}

export interface TargetHints {
  type?: string;
  text_contains?: string[];
  placeholder_contains?: string[];
  selector_pattern?: string;
  role?: string;
}

export interface PlannedStep {
  step_number: number;
  action: 'CLICK' | 'TYPE' | 'SCROLL' | 'WAIT' | 'DONE';
  description: string;
  target_hints: TargetHints;
  text_input?: string;
  expected_page_change: boolean;
}

export interface StartSessionRequest {
  user_goal: string;
  initial_page_features: PageFeature[];
  url: string;
  page_title: string;
}

export interface FirstStepInfo {
  step_number: number;
  action: string;
  target_feature_index: number | null;
  instruction: string;
  confidence: number;
}

export interface StartSessionResponse {
  session_id: string;
  planned_steps: PlannedStep[];
  total_steps: number;
  first_step: FirstStepInfo;
}

export interface NextActionRequest {
  session_id: string;
  page_features: PageFeature[];
  url?: string;
  page_title?: string;
  previous_action_result?: {
    success: boolean;
    error?: string;
  };
}

export interface NextActionResponse {
  step_number: number;
  total_steps: number;
  action: string;
  target_feature_index: number | null;
  target_feature: PageFeature | null;
  instruction: string;
  text_input?: string;
  confidence: number;
  expected_page_change: boolean;
  session_complete: boolean;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Start a new tutoring session - sends goal + page features to backend
 * Backend calls Gemini to generate the workflow plan
 */
export async function startSession(request: StartSessionRequest): Promise<StartSessionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/session/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Get the next action for the current session
 */
export async function getNextAction(request: NextActionRequest): Promise<NextActionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/session/next`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Send correction when user says the suggested element was wrong
 */
export async function sendCorrection(
  sessionId: string,
  feedback: 'wrong_element' | 'doesnt_work',
  actualFeatureIndex?: number
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/session/correct`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      feedback,
      actual_feature_index: actualFeatureIndex,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || `HTTP ${response.status}`);
  }
}

/**
 * Check if the backend is available
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
