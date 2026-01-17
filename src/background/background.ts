// Background Service Worker
console.log('Big Brother: Background service worker loaded');

// Open side panel when extension icon is clicked
chrome.action.onClicked.addListener((tab) => {
  if (tab.id) {
    chrome.sidePanel.open({ tabId: tab.id });
  }
});

// Handle messages from side panel and content scripts
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  console.log('Background received message:', message);

  // Route messages from side panel to content script
  if (message.type === 'USER_PROMPT' && message.target === 'content') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const activeTab = tabs[0];
      
      if (!activeTab?.id) {
        sendResponse({ success: false, error: 'No active tab found' });
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
