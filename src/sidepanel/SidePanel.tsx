import React, { useState, useEffect, useRef } from 'react';
import { sendMessageToContent } from '../utils/messaging';
import { startSession, getNextAction, type PlannedStep, type PageFeature, ApiError } from '../utils/api';
import type { Message, AgentStatus, ContentResponse } from '../types/messages';

interface SessionState {
  sessionId: string | null;
  plannedSteps: PlannedStep[];
  currentStepIndex: number;
  totalSteps: number;
  awaitingConfirmation: boolean; // Are we waiting for user to type "Y"?
  isExecuting: boolean; // Is execution in progress?
  executionError?: string;
}

const SidePanel: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [status, setStatus] = useState<AgentStatus>('idle');
  const [session, setSession] = useState<SessionState | null>(null);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check backend health on mount
  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((res) => setBackendAvailable(res.ok))
      .catch(() => setBackendAvailable(false));
  }, []);

  // Load messages from storage on mount
  useEffect(() => {
    chrome.storage.local.get(['chatHistory', 'sessionState'], (result) => {
      if (result.chatHistory) {
        setMessages(result.chatHistory);
      }
      if (result.sessionState) {
        setSession(result.sessionState);
      }
    });
  }, []);

  // Save messages to storage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      chrome.storage.local.set({ chatHistory: messages });
    }
  }, [messages]);

  // Save session state
  useEffect(() => {
    if (session) {
      chrome.storage.local.set({ sessionState: session });
    }
  }, [session]);

  const addAgentMessage = (content: string, stepInfo?: Message['stepInfo']) => {
    const msg: Message = {
      id: Date.now().toString(),
      role: 'agent',
      content,
      timestamp: new Date().toISOString(),
      stepInfo,
    };
    setMessages((prev) => [...prev, msg]);
  };

  const addUserMessage = (content: string) => {
    const msg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, msg]);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userInput = inputValue.trim();
    
    // Check if we're waiting for confirmation
    if (session?.awaitingConfirmation) {
      addUserMessage(userInput);
      setInputValue('');
      
      if (userInput.toLowerCase() === 'y' || userInput.toLowerCase() === 'yes') {
        // User confirmed - start execution
        addAgentMessage('✅ Great! Starting execution...');
        
        // Update session state - no longer awaiting confirmation, start executing
        setSession(prev => prev ? { ...prev, awaitingConfirmation: false, isExecuting: true } : null);
        
        // Start execution
        executeSteps(session);
        
      } else if (userInput.toLowerCase() === 'n' || userInput.toLowerCase() === 'no') {
        // User declined - cancel the plan
        addAgentMessage('❌ Plan cancelled. Tell me what you\'d like to do instead.');
        // Clear highlights
        sendMessageToContent({ type: 'CLEAR_HIGHLIGHTS', target: 'content' }).catch(() => {});
        setSession(null);
        chrome.storage.local.remove(['sessionState']);
      } else {
        // Invalid input
        addAgentMessage('Please type **Y** to proceed with the plan, or **N** to cancel.');
      }
      return;
    }

    // Normal flow - user is entering a goal
    const userGoal = userInput;
    addUserMessage(userGoal);
    setInputValue('');
    setStatus('thinking');

    try {
      // Step 1: Get page features from content script
      addAgentMessage('🔍 Scanning the page for interactive elements...');
      
      const contentResponse: ContentResponse = await sendMessageToContent({
        type: 'GET_FEATURES',
      });

      if (!contentResponse.success || !contentResponse.features) {
        throw new Error(contentResponse.error || 'Failed to get page features');
      }

      const { features, pageUrl, pageTitle } = contentResponse;
      
      addAgentMessage(`📋 Found **${features.length}** interactive elements on this page.`);

      // Step 2: Send to backend to generate workflow with Gemini
      addAgentMessage('🤖 Asking AI to plan the steps...');
      setStatus('acting');

      const sessionResponse = await startSession({
        user_goal: userGoal,
        initial_page_features: features as PageFeature[],
        url: pageUrl || window.location.href,
        page_title: pageTitle || 'Unknown Page',
      });

      // Store session state - awaiting confirmation
      setSession({
        sessionId: sessionResponse.session_id,
        plannedSteps: sessionResponse.planned_steps,
        currentStepIndex: 0,
        totalSteps: sessionResponse.total_steps,
        awaitingConfirmation: true,
        isExecuting: false,
      });

      // Display the generated plan
      const stepsMessage = formatPlannedSteps(sessionResponse.planned_steps);
      addAgentMessage(
        `📝 **Here's the plan to achieve your goal:**\n\n` +
        `${stepsMessage}\n\n` +
        `---\n` +
        `**Total steps: ${sessionResponse.total_steps}**`
      );

      // Ask for confirmation
      addAgentMessage('👆 **Type Y to proceed with this plan, or N to cancel.**');

      setStatus('idle');
    } catch (error) {
      console.error('Error:', error);
      
      let errorMessage = 'Something went wrong.';
      if (error instanceof ApiError) {
        if (error.status === 503) {
          errorMessage = `⚠️ Backend service error: ${error.message}\n\nMake sure the backend server is running and API keys are configured.`;
        } else {
          errorMessage = `API Error (${error.status}): ${error.message}`;
        }
      } else if (error instanceof Error) {
        if (error.message.includes('Failed to fetch')) {
          errorMessage = '❌ Cannot connect to backend server.\n\nMake sure to start it with:\n```\ncd backend && uvicorn app.main:app --reload\n```';
        } else {
          errorMessage = `Error: ${error.message}`;
        }
      }

      addAgentMessage(errorMessage);
      setStatus('idle');
    }
  };

  const formatPlannedSteps = (steps: PlannedStep[]): string => {
    return steps
      .map((step, idx) => {
        const icon = getActionIcon(step.action);
        const textInput = step.text_input ? `\n   └─ Text: "${step.text_input}"` : '';
        return `**Step ${idx + 1}:** ${icon} ${step.action}\n   ${step.description}${textInput}`;
      })
      .join('\n\n');
  };

  const getActionIcon = (action: string): string => {
    switch (action) {
      case 'CLICK': return '👆';
      case 'TYPE': return '⌨️';
      case 'SCROLL': return '📜';
      case 'WAIT': return '⏳';
      case 'DONE': return '✅';
      default: return '▶️';
    }
  };

  /**
   * Execute all steps in the plan sequentially
   */
  const executeSteps = async (currentSession: SessionState) => {
    if (!currentSession.sessionId || !currentSession.plannedSteps.length) {
      addAgentMessage('❌ No plan to execute.');
      return;
    }

    const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

    const getFeaturesSafe = async (): Promise<ContentResponse> => {
      // Navigation can temporarily unload the content script; retry a few times.
      let lastErr: unknown = null;
      for (let i = 0; i < 20; i++) {
        try {
          const resp: ContentResponse = await sendMessageToContent({
            type: 'GET_FEATURES',
            target: 'content',
          });
          if (resp.success) return resp;
          lastErr = new Error(resp.error || 'Failed to get page features');
        } catch (e) {
          lastErr = e;
        }
        await delay(500);
      }
      throw lastErr instanceof Error ? lastErr : new Error('Failed to get page features');
    };

    const featuresSignature = (features: PageFeature[]): string => {
      // Lightweight fingerprint: enough to notice major DOM shifts.
      return features
        .slice(0, 30)
        .map((f) => `${f.type}|${(f.text || '').slice(0, 50)}|${(f.placeholder || '').slice(0, 30)}|${(f.aria_label || '').slice(0, 30)}|${(f.href || '').slice(0, 50)}`)
        .join('||');
    };

    const waitForPageUpdate = async (opts: {
      prevUrl?: string;
      prevSig?: string;
      expectUrlChange: boolean;
      timeoutMs?: number;
    }): Promise<ContentResponse> => {
      const timeoutMs = opts.timeoutMs ?? 20000;
      const start = Date.now();
      let lastResp: ContentResponse | null = null;
      let stableCount = 0;

      while (Date.now() - start < timeoutMs) {
        await delay(500);
        try {
          const resp = await getFeaturesSafe();
          lastResp = resp;
          const url = resp.pageUrl || '';
          const sig = resp.features ? featuresSignature(resp.features as PageFeature[]) : '';

          if (opts.expectUrlChange && opts.prevUrl && url && url !== opts.prevUrl) {
            // URL changed => treat as page transition complete.
            return resp;
          }

          // If URL didn't change (SPA, modal, etc), wait until the DOM/features stabilize.
          if (!opts.expectUrlChange) {
            if (opts.prevSig && sig && sig !== opts.prevSig) {
              stableCount = 0; // changed, start looking for stability
            } else if (sig) {
              stableCount += 1;
              if (stableCount >= 2) return resp; // stable across ~1s
            }
          } else {
            // expected URL change but didn't happen; accept stabilized DOM as fallback
            if (opts.prevSig && sig && sig !== opts.prevSig) {
              stableCount = 0;
            } else if (sig) {
              stableCount += 1;
              if (stableCount >= 3) return resp; // stable across ~1.5s
            }
          }
        } catch {
          // ignore transient failures during navigation; loop will retry
        }
      }

      // Timeout: return last known state if we have it.
      if (lastResp) return lastResp;
      return getFeaturesSafe();
    };

    setStatus('acting');
    const steps = currentSession.plannedSteps;
    let previousSuccess = true;
    let previousError: string | undefined = undefined;

    try {
      // Drive execution off the backend `/next` so step order stays in sync even with WAIT/SCROLL.
      for (let guard = 0; guard < 200; guard++) {
        // Get current page features (fresh DOM)
        const featuresResponse: ContentResponse = await getFeaturesSafe();
        if (!featuresResponse.features) {
          throw new Error(featuresResponse.error || 'Failed to get page features');
        }
        const prevUrl = featuresResponse.pageUrl || window.location.href;
        const prevSig = featuresSignature(featuresResponse.features as PageFeature[]);

        // Ask backend what the next step is (backend advances only when previousSuccess=true)
        const nextAction = await getNextAction({
          session_id: currentSession.sessionId,
          page_features: featuresResponse.features as PageFeature[],
          url: featuresResponse.pageUrl || window.location.href,
          page_title: featuresResponse.pageTitle || document.title,
          previous_action_result: { success: previousSuccess, error: previousError },
        });

        const stepIndex = Math.max(0, (nextAction.step_number || 1) - 1);
        const step = steps[stepIndex];
        const action = nextAction.action as 'CLICK' | 'TYPE' | 'SCROLL' | 'WAIT' | 'DONE';
        const instruction = nextAction.instruction || step?.description || '';
        const textInput = step?.text_input ?? nextAction.text_input;
        const expectChange = Boolean(nextAction.expected_page_change);

        // DONE / complete
        if (action === 'DONE' || nextAction.session_complete) {
          addAgentMessage('🎉 **All steps completed!** Your goal has been achieved.');
          break;
        }

        // Show current step
        addAgentMessage(
          `\n**Step ${nextAction.step_number}/${nextAction.total_steps}:** ${getActionIcon(action)} ${action}\n` +
          `${instruction}` +
          (textInput ? `\n└─ Text: "${textInput}"` : '')
        );

        // Keep UI session cursor aligned
        setSession((prev) =>
          prev
            ? {
                ...prev,
                currentStepIndex: stepIndex,
                totalSteps: nextAction.total_steps,
              }
            : null
        );

        // No-element actions
        if (action === 'SCROLL') {
          const result = await sendMessageToContent({
            type: 'EXECUTE_ACTION',
            payload: {
              action,
              targetIndex: null,
            },
            target: 'content',
          });

          if (!result.success) {
            addAgentMessage(`⚠️ Action failed: ${result.error}`);
            previousSuccess = false;
            previousError = result.error;
            await delay(1000);
            continue;
          }

          addAgentMessage(`✅ ${result.message || 'Action completed.'}`);
          previousSuccess = true;
          previousError = undefined;
          // Wait for DOM to settle before moving on (SPA / lazy loads)
          await waitForPageUpdate({
            prevUrl,
            prevSig,
            expectUrlChange: expectChange,
            timeoutMs: expectChange ? 20000 : 6000,
          });

          continue;
        }

        // WAIT steps can be "manual": we tell the user what to do and only continue after the page actually updates.
        if (action === 'WAIT') {
          // For manual steps, the planner should set expected_page_change=true.
          // We'll poll for a URL/DOM update; if it doesn't happen, warn and keep the same step.
          addAgentMessage(
            expectChange
              ? '⏳ Waiting for you to finish this step and for the page to update...'
              : '⏳ Waiting...'
          );

          const updated = await waitForPageUpdate({
            prevUrl,
            prevSig,
            expectUrlChange: expectChange,
            timeoutMs: expectChange ? 60000 : 6000,
          });

          const newUrl = updated.pageUrl || '';
          const urlChanged = Boolean(prevUrl && newUrl && newUrl !== prevUrl);
          const domChanged =
            Boolean(updated.features) &&
            featuresSignature(updated.features as PageFeature[]) !== prevSig;

          if (expectChange && !(urlChanged || domChanged)) {
            addAgentMessage(
              '⚠️ I didn’t detect the page changing. Please follow the instruction above (open a new tab / type the URL / press Enter). If you end up on a different site, tell me and I’ll adapt.'
            );
            previousSuccess = false;
            previousError = 'No page change detected after manual WAIT step';
            continue;
          }

          previousSuccess = true;
          previousError = undefined;
          continue;
        }

        if (nextAction.target_feature_index === null) {
          // No matching element found - show what we're looking for
          addAgentMessage(
            `⚠️ **Could not find the right element.**\n` +
            `Looking for: ${instruction}\n` +
            `Try scrolling or waiting for the page to load.`
          );
          
          // Highlight nothing, wait a bit and retry
          await delay(1500);
          continue;
        }

        // Highlight the target element
        addAgentMessage(
          `🎯 Found element: **${nextAction.target_feature?.text || 'Unknown'}** ` +
          `(${nextAction.target_feature?.type}, confidence: ${Math.round(nextAction.confidence * 100)}%)`
        );

        await sendMessageToContent({
          type: 'HIGHLIGHT_ELEMENT',
          payload: {
            targetIndex: nextAction.target_feature_index,
            duration: 3000,
          },
          target: 'content',
        });

        // Wait a moment for user to see the highlight
        await delay(1200);

        // Execute the action
        const actionResult = await sendMessageToContent({
          type: 'EXECUTE_ACTION',
          payload: {
            action: action as 'CLICK' | 'TYPE' | 'SCROLL' | 'WAIT',
            targetIndex: nextAction.target_feature_index,
            textInput,
          },
          target: 'content',
        });

        if (!actionResult.success) {
          addAgentMessage(`⚠️ Action failed: ${actionResult.error}`);
          previousSuccess = false;
          previousError = actionResult.error;
          // Do NOT advance; keep the same step and re-scan the page.
          // This prevents running ahead on stale DOM and allows retry/correction.
          await delay(1000);
          continue;
        } else {
          addAgentMessage(`✅ ${actionResult.message || 'Action completed.'}`);
          previousSuccess = true;
          previousError = undefined;
        }

        // Wait for real completion: URL change when expected, otherwise DOM stabilization.
        if (expectChange) {
          addAgentMessage('⏳ Waiting for page navigation/DOM update...');
        }
        await waitForPageUpdate({
          prevUrl,
          prevSig,
          expectUrlChange: expectChange,
          timeoutMs: expectChange ? 20000 : 6000,
        });

      }

      // Execution complete
      addAgentMessage('\n🎉 **Execution complete!**');
      setSession(prev => prev ? { ...prev, isExecuting: false } : null);
      
    } catch (error) {
      console.error('Execution error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addAgentMessage(`\n❌ **Execution stopped:** ${errorMessage}`);
      setSession(prev => prev ? { ...prev, isExecuting: false, executionError: errorMessage } : null);
    } finally {
      setStatus('idle');
      // Clear highlights when done
      sendMessageToContent({ type: 'CLEAR_HIGHLIGHTS', target: 'content' }).catch(() => {});
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
    setSession(null);
    chrome.storage.local.remove(['chatHistory', 'sessionState']);
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
        return 'Scanning page...';
      case 'acting':
        if (session?.isExecuting) {
          return `Executing step ${session.currentStepIndex + 1}/${session.totalSteps}...`;
        }
        return 'Generating plan...';
      default:
        if (session?.awaitingConfirmation) {
          return 'Awaiting confirmation...';
        }
        return 'Ready';
    }
  };

  const getPlaceholderText = () => {
    if (session?.isExecuting) {
      return 'Execution in progress...';
    }
    if (session?.awaitingConfirmation) {
      return 'Type Y to proceed, or N to cancel';
    }
    return "What do you want to do? (e.g., 'Search for laptops')";
  };

  // Simple markdown-like formatting
  const formatContent = (content: string): React.ReactNode => {
    const renderWithLinks = (text: string, keyPrefix: string): React.ReactNode[] => {
      // Basic URL linkification for http(s) URLs.
      const urlRegex = /(https?:\/\/[^\s<>"']+)/g;
      const parts = text.split(urlRegex);
      return parts.map((p, idx) => {
        const key = `${keyPrefix}-url-${idx}`;
        if (p.match(urlRegex)) {
          return (
            <a
              key={key}
              href={p}
              target="_blank"
              rel="noreferrer noopener"
              className="text-blue-600 underline break-all"
            >
              {p}
            </a>
          );
        }
        return <React.Fragment key={key}>{p}</React.Fragment>;
      });
    };

    // Split by code blocks first
    const parts = content.split(/(```[\s\S]*?```)/g);
    
    return parts.map((part, i) => {
      if (part.startsWith('```')) {
        // Code block
        const code = part.replace(/```\w*\n?/g, '').replace(/```$/g, '');
        return (
          <pre key={i} className="bg-gray-800 text-green-400 p-2 rounded mt-2 mb-2 text-xs overflow-x-auto">
            {code}
          </pre>
        );
      }
      
      // Format bold text and line breaks (+ linkify plain URLs)
      const formatted = part.split(/(\*\*[^*]+\*\*)/g).map((segment, j) => {
        if (segment.startsWith('**') && segment.endsWith('**')) {
          return <strong key={j}>{segment.slice(2, -2)}</strong>;
        }
        // Handle line breaks
        return segment.split('\n').map((line, k, arr) => (
          <React.Fragment key={`${j}-${k}`}>
            {renderWithLinks(line, `${i}-${j}-${k}`)}
            {k < arr.length - 1 && <br />}
          </React.Fragment>
        ));
      });
      
      return <span key={i}>{formatted}</span>;
    });
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 shadow-sm">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-800">Big Bro</h1>
          <button
            onClick={clearHistory}
            className="text-sm text-gray-600 hover:text-gray-800 px-3 py-1 rounded hover:bg-gray-100 transition-colors"
          >
            Clear
          </button>
        </div>
        
        {/* Status Indicator */}
        <div className="flex items-center gap-2 mt-2">
          <div className={`w-2 h-2 rounded-full ${getStatusColor()} ${status !== 'idle' ? 'animate-pulse' : ''}`} />
          <span className="text-sm text-gray-600">{getStatusText()}</span>
          {backendAvailable === false && (
            <span className="text-xs text-red-500 ml-2">⚠️ Backend offline</span>
          )}
          {backendAvailable === true && (
            <span className="text-xs text-green-600 ml-2">✓ Connected</span>
          )}
        </div>

        {/* Session info */}
        {session && (
          <div className="mt-2 text-xs text-gray-500">
            Session: {session.sessionId?.slice(0, 8)}... | Step: {session.currentStepIndex + 1}/{session.totalSteps}
            {session.awaitingConfirmation && (
              <span className="ml-2 text-orange-500 font-medium">⏳ Awaiting Y/N</span>
            )}
            {session.isExecuting && (
              <span className="ml-2 text-blue-500 font-medium">▶️ Executing</span>
            )}
          </div>
        )}
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <p className="text-lg mb-2">👋 Welcome!</p>
              <p className="text-sm">Tell me what you want to do on this page.</p>
              <p className="text-xs mt-2 text-gray-400">
                e.g., "Search for wireless headphones" or "Sign up for an account"
              </p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
                }`}
              >
                <div className="text-sm whitespace-pre-wrap break-words">
                  {formatContent(message.content)}
                </div>
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
            placeholder={getPlaceholderText()}
            disabled={status !== 'idle' || session?.isExecuting}
            className={`flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 disabled:bg-gray-100 disabled:cursor-not-allowed ${
              session?.isExecuting
                ? 'border-blue-400 focus:ring-blue-500'
                : session?.awaitingConfirmation 
                  ? 'border-orange-400 focus:ring-orange-500' 
                  : 'border-gray-300 focus:ring-blue-500'
            }`}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || status !== 'idle' || session?.isExecuting}
            className={`px-6 py-2 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium ${
              session?.isExecuting
                ? 'bg-blue-500'
                : session?.awaitingConfirmation
                  ? 'bg-orange-500 hover:bg-orange-600'
                  : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {session?.isExecuting ? '▶️' : session?.awaitingConfirmation ? 'Confirm' : 'Go'}
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          {session?.isExecuting
            ? `Executing step ${session.currentStepIndex + 1} of ${session.totalSteps}...`
            : session?.awaitingConfirmation 
              ? 'Type Y and press Enter to start, or N to cancel'
              : 'Press Enter to send • AI will generate step-by-step instructions'
          }
        </p>
      </div>
    </div>
  );
};

export default SidePanel;
