import React, { useState, useEffect, useRef } from 'react';
import { sendMessageToContent } from '../utils/messaging';
import { startSession, getNextAction, type PlannedStep, type PageFeature, ApiError } from '../utils/api';
import type { Message, AgentStatus, ContentResponse } from '../types/messages';
import HealthySnacks from './HealthySnacks';

type TabType = 'assistant' | 'snacks';

interface SessionState {
  sessionId: string | null;
  plannedSteps: PlannedStep[];
  currentStepIndex: number;
  totalSteps: number;
  awaitingConfirmation: boolean; // Are we waiting for user to type "Y"?
  isExecuting: boolean; // Is execution in progress?
  executionError?: string;
}

// Extend Window interface for Web Speech API
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

const SidePanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('assistant');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [status, setStatus] = useState<AgentStatus>('idle');
  const [session, setSession] = useState<SessionState | null>(null);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const manualAdvanceRef = useRef<boolean>(false);
  const manualAdvanceNoteRef = useRef<string>('');
  const stopExecutionRef = useRef<boolean>(false);
  const [speechOutputEnabled, setSpeechOutputEnabled] = useState(false);
  const [showMicInstructions, setShowMicInstructions] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [micPermissionGranted, setMicPermissionGranted] = useState<boolean | null>(null);
  const spokenMessageIdsRef = useRef<Set<string>>(new Set());
  const recognitionRef = useRef<any>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

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

  // Check microphone permission on mount
  useEffect(() => {
    checkMicrophonePermission();
    
    // Check periodically in case permission is granted from settings page
    const interval = setInterval(checkMicrophonePermission, 2000);
    return () => clearInterval(interval);
  }, []);

  // Initialize speech recognition if permission is granted
  useEffect(() => {
    if (micPermissionGranted === true) {
      initializeSpeechRecognition();
    }
  }, [micPermissionGranted]);

  const checkMicrophonePermission = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setMicPermissionGranted(false);
        return;
      }

      // Try to access microphone to check permission
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      
      // Permission is granted
      setMicPermissionGranted(true);
      setShowMicInstructions(false);
    } catch (error: any) {
      // Permission is not granted
      setMicPermissionGranted(false);
    }
  };

  const initializeSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputValue(transcript);
        setIsListening(false);
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        
        if (event.error === 'not-allowed') {
          setMicPermissionGranted(false);
          setShowMicInstructions(true);
        }
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  };

  // ElevenLabs text-to-speech function
  const speakWithElevenLabs = async (text: string, messageId: string) => {
    // Skip if already spoken
    if (spokenMessageIdsRef.current.has(messageId)) {
      console.log('TTS: Message already spoken, skipping:', messageId);
      return;
    }

    // Stop any currently playing audio
    if (currentAudioRef.current) {
      console.log('TTS: Stopping previous audio');
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      if (currentAudioRef.current.src) {
        URL.revokeObjectURL(currentAudioRef.current.src);
      }
      currentAudioRef.current = null;
    }

    try {
      // Get API key from environment variable
      const apiKey = import.meta.env.VITE_ELEVENLABS_API_KEY;
      const voiceId = import.meta.env.VITE_ELEVENLABS_VOICE_ID || '21m00Tcm4TlvDq8ikWAM'; // Default voice: Rachel

      console.log('ElevenLabs TTS - API Key present:', !!apiKey, 'Voice ID:', voiceId);
      
      if (!apiKey || apiKey === 'your_elevenlabs_api_key_here') {
        console.warn('ElevenLabs API key not found. Please set VITE_ELEVENLABS_API_KEY in your .env file.');
        console.warn('Current API key value:', apiKey ? 'Set (but may be placeholder)' : 'Not set');
        return;
      }

      console.log('ElevenLabs TTS - Making API call for text:', text.substring(0, 50) + '...');

      const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': apiKey,
        },
        body: JSON.stringify({
          text: text,
          model_id: 'eleven_turbo_v2_5', // Updated to newer model available on free tier
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.5,
          },
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `ElevenLabs API error: ${response.status} ${response.statusText}`;
        try {
          const errorJson = JSON.parse(errorText);
          if (errorJson.detail?.message) {
            errorMessage = `ElevenLabs API error: ${errorJson.detail.message}`;
          }
        } catch {
          // If JSON parsing fails, use the text as is
          if (errorText) {
            errorMessage = `ElevenLabs API error: ${errorText}`;
          }
        }
        throw new Error(errorMessage);
      }

      const audioBlob = await response.blob();
      console.log('TTS: Audio blob received, size:', audioBlob.size, 'bytes');
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      // Store reference to current audio for stopping
      currentAudioRef.current = audio;
      
      // Set volume to ensure it's audible
      audio.volume = 1.0;

      console.log('TTS: Attempting to play audio...');
      await new Promise((resolve, reject) => {
        audio.onended = () => {
          console.log('TTS: Audio playback completed');
          URL.revokeObjectURL(audioUrl);
          currentAudioRef.current = null;
          resolve(undefined);
        };
        audio.onerror = (e) => {
          console.error('TTS: Audio playback error:', e);
          URL.revokeObjectURL(audioUrl);
          currentAudioRef.current = null;
          reject(new Error('Audio playback failed'));
        };
        audio.oncanplay = () => {
          console.log('TTS: Audio can play');
        };
        
        const playPromise = audio.play();
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              console.log('TTS: Audio started playing successfully');
            })
            .catch((error) => {
              console.error('TTS: Audio play() promise rejected:', error);
              reject(error);
            });
        } else {
          console.log('TTS: Audio play() returned undefined');
        }
      });

      // Mark as spoken
      spokenMessageIdsRef.current.add(messageId);
    } catch (error: any) {
      console.error('Error with ElevenLabs TTS:', error);
      // Log more details for debugging
      if (error.message) {
        console.error('ElevenLabs error details:', error.message);
      }
    }
  };

  // Text-to-speech for new agent messages
  useEffect(() => {
    console.log('TTS Effect triggered - speechOutputEnabled:', speechOutputEnabled, 'messages.length:', messages.length);
    
    if (!speechOutputEnabled) {
      console.log('TTS: Speech output is disabled');
      return;
    }
    
    if (messages.length === 0) {
      console.log('TTS: No messages to speak');
      return;
    }

    const lastMessage = messages[messages.length - 1];
    console.log('TTS: Last message:', { role: lastMessage.role, id: lastMessage.id, alreadySpoken: spokenMessageIdsRef.current.has(lastMessage.id) });
    
    // Only speak new agent messages that haven't been spoken yet
    if (lastMessage.role === 'agent' && !spokenMessageIdsRef.current.has(lastMessage.id)) {
      console.log('TTS: Speaking new agent message:', lastMessage.id, 'Content:', lastMessage.content.substring(0, 50));
      speakWithElevenLabs(lastMessage.content, lastMessage.id);
    } else {
      console.log('TTS: Skipping message - not agent or already spoken');
    }
  }, [messages, speechOutputEnabled]);

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
    const consumeManualAdvance = (): { advanced: boolean; note?: string } => {
      if (!manualAdvanceRef.current) return { advanced: false };
      manualAdvanceRef.current = false;
      const note = manualAdvanceNoteRef.current;
      manualAdvanceNoteRef.current = '';
      return { advanced: true, note };
    };

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
    // IMPORTANT: start false so backend doesn't advance to step 2 before the user completes step 1.
    let previousSuccess = false;
    let previousError: string | undefined = undefined;

    // Reset stop flag at the start of execution
    stopExecutionRef.current = false;

    // Track recent actions to detect loops
    const recentActions: Array<{ action: string; targetText: string; url: string }> = [];

    try {
      const waitForManualPageChange = async (opts: {
        prevUrl: string;
        prevSig: string;
        timeoutMs: number;
        remindEveryMs: number;
      }): Promise<boolean> => {
        const start = Date.now();
        let lastRemind = 0;
        while (Date.now() - start < opts.timeoutMs) {
          const manual = consumeManualAdvance();
          if (manual.advanced) return true;
          await delay(1000);
          const resp = await getFeaturesSafe();
          const newUrl = resp.pageUrl || '';
          const newSig = resp.features ? featuresSignature(resp.features as PageFeature[]) : '';
          const urlChanged = Boolean(opts.prevUrl && newUrl && newUrl !== opts.prevUrl);
          const domChanged = Boolean(newSig && newSig !== opts.prevSig);

          if (urlChanged || domChanged) return true;

          if (Date.now() - lastRemind >= opts.remindEveryMs) {
            lastRemind = Date.now();
            addAgentMessage('⏳ Still waiting — please complete the step above.');
          }
        }
        return false;
      };

      // Drive execution off the backend `/next` so step order stays in sync.
      for (let guard = 0; guard < 200; guard++) {
        // Check if user wants to stop execution
        if (stopExecutionRef.current) {
          addAgentMessage('🛑 Execution stopped by user.');
          break;
        }

        // Get current page features (fresh DOM)
        const featuresResponse: ContentResponse = await getFeaturesSafe();
        if (!featuresResponse.features) {
          throw new Error(featuresResponse.error || 'Failed to get page features');
        }

        // Debug: log what the extension sees from the page (open the Side Panel DevTools console)
        try {
          console.groupCollapsed(
            '[Big Brother][SidePanel] GET_FEATURES',
            featuresResponse.pageUrl || window.location.href
          );
          console.table(
            (featuresResponse.features as PageFeature[]).slice(0, 30).map((f) => ({
              index: f.index,
              type: f.type,
              text: f.text,
              placeholder: f.placeholder || '',
              aria_label: f.aria_label || '',
              href: f.href || '',
              value_len: f.value_len ?? 0,
              selector: f.selector,
            }))
          );
          console.groupEnd();
        } catch {
          // ignore console.table issues
        }
        const prevUrl = featuresResponse.pageUrl || window.location.href;
        const prevSig = featuresSignature(featuresResponse.features as PageFeature[]);

        // Manual override: if user pressed "Next step", treat previous step as complete.
        const manualBeforeNext = consumeManualAdvance();
        if (manualBeforeNext.advanced) {
          previousSuccess = true;
          previousError = manualBeforeNext.note || 'Manual advance';
        }

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

        // Guidance-only: SCROLL is manual.
        if (action === 'SCROLL') {
          addAgentMessage('Please scroll down a bit on the page.');
          let detected = false;
          for (let i = 0; i < 12; i++) {
            const manual = consumeManualAdvance();
            if (manual.advanced) {
              detected = true;
              break;
            }
            const waited = await sendMessageToContent({
              type: 'WAIT_FOR_EVENT',
              payload: { event: 'scroll', timeoutMs: 2500 },
              target: 'content',
            });
            if (waited.success) {
              detected = true;
              break;
            }
          }
          if (!detected) {
            addAgentMessage(`⚠️ I didn't detect scrolling yet. You can scroll again, or press **Next step** if you already did it.`);
            previousSuccess = false;
            previousError = 'No scroll detected';
            await delay(3000);
            continue;
          }
          previousSuccess = true;
          previousError = undefined;
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
          if (expectChange) {
            const changed = await waitForManualPageChange({
              prevUrl,
              prevSig,
              timeoutMs: 60000,
              remindEveryMs: 15000,
            });
            if (!changed) {
              addAgentMessage(
                '⚠️ I still didn’t detect the page changing. Please do the step above. If you went to the wrong site, go back and open:\nhttps://www.instagram.com/accounts/emailsignup/'
              );
              previousSuccess = false;
              previousError = 'No page change detected after manual WAIT step';
              await delay(5000);
              continue;
            }
          } else {
            // Short wait, but still allow manual override.
            for (let i = 0; i < 2; i++) {
              const manual = consumeManualAdvance();
              if (manual.advanced) break;
              await delay(1000);
            }
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
            targetIndex: nextAction.target_feature_index ?? undefined,
            selector: nextAction.target_feature?.selector,
            // Keep the highlight until the next step (or until we clear highlights).
            duration: 0,
          },
          target: 'content',
        });

        // Wait a moment for user to see the highlight
        await delay(1200);

        // Track this action (no warning needed - AI already sees already_clicked flag)
        recentActions.push({
          action,
          targetText: nextAction.target_feature?.text || '',
          url: prevUrl,
        });
        
        // Keep only last 10 actions in memory
        if (recentActions.length > 10) {
          recentActions.shift();
        }

        // Guidance-only: user performs the action. We just wait for the event.
        if (action === 'CLICK') {
          // Automatically execute the CLICK action
          addAgentMessage('🖱️ Clicking the element...');
          
          const executeResult = await sendMessageToContent({
            type: 'EXECUTE_ACTION',
            payload: {
              action: 'CLICK',
              targetIndex: nextAction.target_feature_index ?? null,
            },
            target: 'content',
          });

          if (!executeResult.success) {
            addAgentMessage(`❌ Failed to click: ${executeResult.error || 'Unknown error'}`);
            previousSuccess = false;
            previousError = executeResult.error || 'Click action failed';
            await delay(3000);
            continue;
          }

          addAgentMessage(`✅ Successfully clicked the element.`);
          previousSuccess = true;
          previousError = undefined;

          // Wait for page update if expected
          if (expectChange) {
            await waitForPageUpdate({
              prevUrl,
              prevSig,
              expectUrlChange: true,
              timeoutMs: 20000,
            });
          }
        } else if (action === 'TYPE') {
          // Automatically execute the TYPE action
          addAgentMessage(`⌨️ Typing "${textInput || ''}" into the field...`);
          
          const executeResult = await sendMessageToContent({
            type: 'EXECUTE_ACTION',
            payload: {
              action: 'TYPE',
              targetIndex: nextAction.target_feature_index ?? null,
              textInput: textInput || '',
            },
            target: 'content',
          });

          if (!executeResult.success) {
            addAgentMessage(`❌ Failed to type: ${executeResult.error || 'Unknown error'}`);
            previousSuccess = false;
            previousError = executeResult.error || 'Type action failed';
            await delay(3000);
            continue;
          }

          addAgentMessage(`✅ Successfully typed into the field.`);
          previousSuccess = true;
          previousError = undefined;
          previousSuccess = true;
          previousError = undefined;

          if (expectChange) {
            await waitForPageUpdate({
              prevUrl,
              prevSig,
              expectUrlChange: true,
              timeoutMs: 20000,
            });
          }
        } else {
          // For other actions (SCROLL, WAIT with element, etc.), use guidance mode
          addAgentMessage(
            `👆 **Please ${(action as string).toLowerCase()} this element:**\n` +
            `${nextAction.target_feature?.text || instruction}`
          );

          const eventReceived = await sendMessageToContent({
            type: 'WAIT_FOR_EVENT',
            payload: {
              action,
              targetIndex: nextAction.target_feature_index ?? undefined,
              selector: nextAction.target_feature?.selector,
            },
            target: 'content',
          });

          if (!eventReceived.success) {
            addAgentMessage('⚠️ No interaction detected. Moving on...');
            previousSuccess = false;
            previousError = 'User interaction timeout';
          } else {
            addAgentMessage('✅ Action completed!');
            previousSuccess = true;
            previousError = undefined;
          }

          if (expectChange) {
            await waitForPageUpdate({
              prevUrl,
              prevSig,
              expectUrlChange: true,
              timeoutMs: 20000,
            });
          }
        }

        // Wait for real completion: URL change when expected, otherwise DOM stabilization.
        // (If we already waited above for expectChange on CLICK/TYPE, this becomes a quick settle.)
        await waitForPageUpdate({
          prevUrl,
          prevSig,
          expectUrlChange: false,
          timeoutMs: 3000,
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
    spokenMessageIdsRef.current.clear();
  };

  const stopExecution = () => {
    stopExecutionRef.current = true;
    addAgentMessage('🛑 **Execution stopped by user.**');
    setSession((prev) => prev ? { ...prev, isExecuting: false } : null);
    setStatus('idle');
  };

  const toggleSpeechInput = async () => {
    // Check permission first
    await checkMicrophonePermission();

    if (micPermissionGranted !== true) {
      // Permission not granted - redirect to settings and show instructions
      const extensionId = chrome.runtime.id;
      const settingsUrl = `chrome://settings/content/siteDetails?site=chrome-extension://${extensionId}`;
      chrome.tabs.create({ url: settingsUrl });
      setShowMicInstructions(true);
      return;
    }

    // Permission is granted - use speech recognition
    if (!recognitionRef.current) {
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      try {
        recognitionRef.current.start();
      } catch (error) {
        console.error('Error starting speech recognition:', error);
        setIsListening(false);
      }
    }
  };

  const toggleSpeechOutput = () => {
    const newState = !speechOutputEnabled;
    console.log('TTS: Toggling speech output to:', newState);
    
    // Stop any currently playing audio when toggling off
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
      currentAudioRef.current = null;
    }
    
    setSpeechOutputEnabled(newState);
    // Clear spoken messages when toggling off so they can be re-spoken if toggled back on
    if (!newState) {
      spokenMessageIdsRef.current.clear();
      console.log('TTS: Cleared spoken messages history');
    } else {
      console.log('TTS: Speech output enabled - will speak new agent messages');
    }
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

  // Handle buying a product from the snacks tab
  const handleBuyProduct = async (product: { title: string; vendor?: string; price: number; url?: string }) => {
    // Switch to assistant tab
    setActiveTab('assistant');
    
    // Clear previous session
    setSession(null);
    chrome.storage.local.remove(['sessionState']);
    
    // Add message explaining what we're doing
    addAgentMessage(
      `🛒 **Starting autonomous purchase flow**\n\n` +
      `Product: **${product.title}**\n` +
      `Price: $${product.price}\n\n` +
      `Navigating to the store...`
    );
    
    // Step 1: Navigate to the product/store URL
    const storeUrl = product.url || `https://www.amazon.ca/s?k=${encodeURIComponent(product.title)}`;
    
    try {
      // Get the active tab and navigate
      const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (activeTab?.id) {
        await chrome.tabs.update(activeTab.id, { url: storeUrl });
        
        // Wait for page to load
        addAgentMessage(`📍 Navigating to: ${new URL(storeUrl).hostname}...`);
        
        // Wait 3 seconds for page load, then start the assistant
        setTimeout(() => {
          const buyGoal = `Find "${product.title}" on this page and add it to cart. The product should be around $${product.price}.`;
          setInputValue(buyGoal);
          
          addAgentMessage('🔍 Page loaded! Now searching for the product...');
          
          // Trigger the buy flow
          setTimeout(() => {
            handleBuyGoal(buyGoal);
          }, 500);
        }, 3000);
      }
    } catch (error) {
      console.error('Navigation error:', error);
      addAgentMessage(`❌ Could not navigate to store. Please go to ${storeUrl} manually.`);
    }
  };

  const handleBuyGoal = async (buyGoal: string) => {
    addUserMessage(buyGoal);
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

      // Step 2: Send to backend to generate workflow
      addAgentMessage('🤖 Planning the purchase steps...');
      setStatus('acting');

      const sessionResponse = await startSession({
        user_goal: buyGoal,
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
        `📝 **Here's my plan to buy this for you:**\n\n` +
        `${stepsMessage}\n\n` +
        `---\n` +
        `**Total steps: ${sessionResponse.total_steps}**`
      );

      // Ask for confirmation
      addAgentMessage('👆 **Type Y to proceed with the purchase, or N to cancel.**');

      setStatus('idle');
    } catch (error) {
      console.error('Error:', error);
      
      let errorMessage = 'Something went wrong.';
      if (error instanceof ApiError) {
        errorMessage = `API Error: ${error.message}`;
      } else if (error instanceof Error) {
        errorMessage = `Error: ${error.message}`;
      }

      addAgentMessage(errorMessage);
      setStatus('idle');
    }
  };

  // If snacks tab is active, render the HealthySnacks component
  if (activeTab === 'snacks') {
    return (
      <div className="flex flex-col h-screen bg-gray-50">
        {/* Tab Bar */}
        <div className="bg-white border-b flex">
          <button
            onClick={() => setActiveTab('assistant')}
            className="flex-1 py-3 text-sm font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-50 transition-colors border-b-2 border-transparent"
          >
            🤖 Assistant
          </button>
          <button
            onClick={() => setActiveTab('snacks')}
            className="flex-1 py-3 text-sm font-medium text-green-600 border-b-2 border-green-600 bg-green-50"
          >
            🥗 Healthy Snacks
          </button>
        </div>
        <div className="flex-1 overflow-hidden">
          <HealthySnacks onBuyProduct={handleBuyProduct} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Tab Bar */}
      <div className="bg-white border-b flex">
        <button
          onClick={() => setActiveTab('assistant')}
          className="flex-1 py-3 text-sm font-medium text-blue-600 border-b-2 border-blue-600 bg-blue-50"
        >
          🤖 Assistant
        </button>
        <button
          onClick={() => setActiveTab('snacks')}
          className="flex-1 py-3 text-sm font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-50 transition-colors border-b-2 border-transparent"
        >
          🥗 Healthy Snacks
        </button>
      </div>

      {/* Microphone Instructions Banner */}
      {showMicInstructions && (
        <div className="bg-orange-500 text-white px-4 py-4 border-b-2 border-orange-600">
          <div className="flex items-start gap-3">
            <div className="text-2xl">🎤</div>
            <div className="flex-1">
              <h3 className="font-bold text-lg mb-2">Enable Microphone Access</h3>
              <p className="text-sm mb-3">
                To use speech-to-text, you need to enable microphone permissions for this extension.
              </p>
              <ol className="text-sm list-decimal list-inside space-y-1 mb-3">
                <li>Go to the Chrome settings tab that just opened</li>
                <li>Find "Microphone" in the permissions list</li>
                <li>Change it from "Block" to "Allow"</li>
                <li>Come back to this extension</li>
                <li>The microphone will automatically be enabled!</li>
              </ol>
              <button
                onClick={() => setShowMicInstructions(false)}
                className="text-sm underline hover:no-underline"
              >
                Dismiss (will check automatically)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-3 shadow-sm">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-800">Big Brother</h1>
          <div className="flex items-center gap-2">
            {session?.isExecuting && (
              <button
                onClick={() => {
                  manualAdvanceRef.current = true;
                  manualAdvanceNoteRef.current = 'Manual advance button pressed';
                  addAgentMessage('➡️ Manual advance requested. Moving to the next step.');
                }}
                className="text-sm text-white bg-orange-500 hover:bg-orange-600 px-3 py-1 rounded transition-colors"
                title="If the app didn't detect your click/typing, press this to continue."
              >
                Next step
              </button>
            )}
            <button
              onClick={toggleSpeechOutput}
              className={`text-sm px-3 py-1 rounded transition-colors ${
                speechOutputEnabled
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
              }`}
              title="Toggle speech output (ElevenLabs TTS)"
            >
              🔊 {speechOutputEnabled ? 'On' : 'Off'}
            </button>
            <button
              onClick={clearHistory}
              className="text-sm text-gray-600 hover:text-gray-800 px-3 py-1 rounded hover:bg-gray-100 transition-colors"
            >
              Clear History
            </button>
          </div>
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
          {session?.isExecuting && (
            <button
              onClick={stopExecution}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
              title="Stop the current execution"
            >
              🛑 Stop
            </button>
          )}
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
            onClick={toggleSpeechInput}
            disabled={status !== 'idle'}
            className={`px-4 py-2 rounded-lg transition-colors font-medium ${
              isListening
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            } disabled:bg-gray-100 disabled:cursor-not-allowed`}
            title={micPermissionGranted ? "Speech to text input" : "Enable microphone access"}
          >
            🎤
          </button>
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
          {isListening && <span className="text-red-600 ml-2">● Listening...</span>}
        </p>
      </div>

    </div>
  );
};

export default SidePanel;
