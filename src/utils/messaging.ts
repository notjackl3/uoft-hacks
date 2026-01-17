import type { ChromeMessage, ContentResponse } from '../types/messages';

/**
 * Send a message to the content script via the background service worker
 */
export async function sendMessageToContent(
  message: ChromeMessage
): Promise<ContentResponse> {
  return new Promise((resolve, reject) => {
    // Add target to route through background to content
    const messageWithTarget = {
      ...message,
      target: 'content' as const,
    };

    chrome.runtime.sendMessage(messageWithTarget, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(response);
      }
    });
  });
}

/**
 * Send a message to the background script
 */
export async function sendMessageToBackground(
  message: ChromeMessage
): Promise<any> {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (response) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(response);
      }
    });
  });
}

/**
 * Get the current active tab
 */
export async function getActiveTab(): Promise<chrome.tabs.Tab | null> {
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      resolve(tabs[0] || null);
    });
  });
}
