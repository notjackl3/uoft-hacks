// Content Script - Injected into every page
console.log('Big Brother: Content script loaded on', window.location.href);

// Listen for messages from the side panel (via background script)
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  console.log('Content script received message:', message);

  if (message.type === 'USER_PROMPT') {
    handleUserPrompt(message.payload.prompt)
      .then((result) => {
        sendResponse({ success: true, ...result });
      })
      .catch((error) => {
        console.error('Error handling user prompt:', error);
        sendResponse({ success: false, error: error.message });
      });

    // Return true to indicate we'll send response asynchronously
    return true;
  }

  return false;
});

/**
 * Handle user prompts and interact with the DOM
 */
async function handleUserPrompt(prompt: string): Promise<{ message: string; domSnapshot?: string }> {
  console.log('Processing prompt:', prompt);

  // Example: Log the DOM body to prove we have access
  const bodyText = document.body.innerText.substring(0, 500); // First 500 chars
  const pageTitle = document.title;
  const pageUrl = window.location.href;

  console.log('Current page title:', pageTitle);
  console.log('Current page URL:', pageUrl);
  console.log('Body text preview:', bodyText);

  // Get DOM statistics
  const domStats = {
    totalElements: document.querySelectorAll('*').length,
    links: document.querySelectorAll('a').length,
    images: document.querySelectorAll('img').length,
    buttons: document.querySelectorAll('button').length,
    inputs: document.querySelectorAll('input').length,
  };

  console.log('DOM Statistics:', domStats);

  // Build response message
  const message = `
 **Page Inspected**: ${pageTitle}
 **URL**: ${pageUrl}

**DOM Statistics**:
- Total Elements: ${domStats.totalElements}
- Links: ${domStats.links}
- Images: ${domStats.images}
- Buttons: ${domStats.buttons}
- Input Fields: ${domStats.inputs}

**Your Prompt**: "${prompt}"

I have access to the DOM and can interact with this page. Ready for LLM integration!
  `.trim();

  return {
    message,
    domSnapshot: bodyText,
  };
}

/**
 * Helper function to query DOM elements
 */
function queryDOMElements(selector: string): Element[] {
  try {
    return Array.from(document.querySelectorAll(selector));
  } catch (error) {
    console.error('Error querying DOM:', error);
    return [];
  }
}

/**
 * Helper function to click an element
 */
function clickElement(selector: string): boolean {
  try {
    const element = document.querySelector(selector) as HTMLElement;
    if (element) {
      element.click();
      return true;
    }
    return false;
  } catch (error) {
    console.error('Error clicking element:', error);
    return false;
  }
}

/**
 * Helper function to fill an input field
 */
function fillInput(selector: string, value: string): boolean {
  try {
    const element = document.querySelector(selector) as HTMLInputElement;
    if (element) {
      element.value = value;
      element.dispatchEvent(new Event('input', { bubbles: true }));
      element.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    }
    return false;
  } catch (error) {
    console.error('Error filling input:', error);
    return false;
  }
}

// Export helper functions for potential future use
export { queryDOMElements, clickElement, fillInput };
