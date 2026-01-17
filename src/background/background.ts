// Background Service Worker
console.log('Big Brother: Background service worker loaded');

// Open side panel when extension icon is clicked
chrome.action.onClicked.addListener((tab) => {
  if (tab.id) {
    chrome.sidePanel.open({ tabId: tab.id });
  }
});

/**
 * Inject content script into a tab if not already present
 */
async function ensureContentScript(tabId: number): Promise<boolean> {
  try {
    // Try to ping the content script
    await chrome.tabs.sendMessage(tabId, { type: 'PING' });
    return true; // Content script is already loaded
  } catch {
    // Content script not loaded, inject it
    console.log('Content script not found, injecting...');
    try {
      // Use whatever the manifest says the content script entrypoint is.
      // This avoids hardcoding dev-only paths like `src/content/content.js`,
      // and works after build when the bundler rewrites filenames.
      const manifest = chrome.runtime.getManifest();
      const files =
        (manifest.content_scripts?.[0]?.js as string[] | undefined) ?? [];
      if (!files.length) {
        console.error('No content_scripts entry found in manifest; cannot inject.');
        return false;
      }
      await chrome.scripting.executeScript({
        target: { tabId },
        files,
      });
      // Wait a bit for the script to initialize
      await new Promise((resolve) => setTimeout(resolve, 100));
      console.log('Content script injected successfully');
      return true;
    } catch (err) {
      console.error('Failed to inject content script:', err);
      return false;
    }
  }
}

// Handle messages from side panel and content scripts
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  console.log('Background received message:', message);

  // Route messages from side panel to content script
  const contentMessageTypes = ['USER_PROMPT', 'GET_FEATURES', 'EXECUTE_ACTION', 'HIGHLIGHT_ELEMENT', 'CLEAR_HIGHLIGHTS'];
  
  if (contentMessageTypes.includes(message.type) && message.target === 'content') {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const activeTab = tabs[0];
      
      if (!activeTab?.id) {
        sendResponse({ success: false, error: 'No active tab found' });
        return;
      }

      // Ensure content script is loaded
      const injected = await ensureContentScript(activeTab.id);
      if (!injected) {
        sendResponse({ 
          success: false, 
          error: 'Could not inject content script. Try refreshing the page.' 
        });
        return;
      }

      // Send message to content script in active tab
      chrome.tabs.sendMessage(
        activeTab.id,
        message,
        (response) => {
          if (chrome.runtime.lastError) {
            console.error('Error sending to content script:', chrome.runtime.lastError);
            sendResponse({ 
              success: false, 
              error: chrome.runtime.lastError.message 
            });
          } else {
            sendResponse(response);
          }
        }
      );
    });

    // Return true to indicate we'll send response asynchronously
    return true;
  }

  // Handle other message types here
  return true;
});

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('Big Brother installed successfully');
  } else if (details.reason === 'update') {
    console.log('Big Brother updated to version', chrome.runtime.getManifest().version);
  }
});
