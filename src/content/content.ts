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
async function handleUserPrompt(prompt: string): Promise<{ message: string; elements?: any[] }> {
  console.log('Processing prompt:', prompt);

  const pageTitle = document.title;
  const pageUrl = window.location.href;

  // Collect all interactive elements
  const interactiveElements: any[] = [];
  let index = 0;

  // Collect links
  const links = Array.from(document.querySelectorAll('a'));
  links.slice(0, 50).forEach((link) => {
    const text = link.innerText.trim() || link.textContent?.trim() || '';
    const href = link.getAttribute('href') || '';
    if (text || href) {
      const groupInfo = getGroupInfo(link);
      interactiveElements.push({
        index: index++,
        type: 'link',
        text: text.substring(0, 100),
        selector: generateSelector(link),
        href,
        group: groupInfo.id,
        groupLabel: groupInfo.label
      });
    }
  });

  // Collect buttons
  const buttons = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"]'));
  buttons.slice(0, 30).forEach((button) => {
    const text = button.textContent?.trim() || (button as HTMLInputElement).value || '';
    if (text) {
      const groupInfo = getGroupInfo(button);
      interactiveElements.push({
        index: index++,
        type: 'button',
        text: text.substring(0, 100),
        selector: generateSelector(button),
        group: groupInfo.id,
        groupLabel: groupInfo.label
      });
    }
  });

  // Collect inputs
  const inputs = Array.from(document.querySelectorAll('input:not([type="button"]):not([type="submit"]), textarea'));
  inputs.slice(0, 20).forEach((input) => {
    const placeholder = input.getAttribute('placeholder') || '';
    const name = input.getAttribute('name') || '';
    const type = input.getAttribute('type') || 'text';
    const groupInfo = getGroupInfo(input);
    interactiveElements.push({
      index: index++,
      type: 'input',
      text: placeholder || name || type,
      selector: generateSelector(input),
      group: groupInfo.id,
      groupLabel: groupInfo.label
    });
  });

  console.log(`Found ${interactiveElements.length} interactive elements (before deduplication)`);

  // Deduplicate based on text + href/type combination
  const seen = new Set<string>();
  const uniqueElements = interactiveElements.filter((el) => {
    const key = `${el.type}:${el.text}:${el.href || ''}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });

  // Reindex after deduplication
  uniqueElements.forEach((el, idx) => {
    el.index = idx;
  });

  console.log(`After deduplication: ${uniqueElements.length} unique elements`);

  // Highlight a random link or button for testing (no inputs)
  let highlightedElement = null;
  const linksAndButtons = uniqueElements.filter(e => e.type === 'link' || e.type === 'button');
  
  if (linksAndButtons.length > 0) {
    const randomIndex = Math.floor(Math.random() * linksAndButtons.length);
    highlightedElement = linksAndButtons[randomIndex];
    
    let element = getElementBySelector(highlightedElement.selector);
    if (element) {
      console.log(`Highlighting ${highlightedElement.type} #${highlightedElement.index}: ${highlightedElement.text}`);
      highlightElement(element);
      movePointerToElement(element);
      
      // Click the element after a short delay
      setTimeout(() => {
        console.log(`Clicking ${highlightedElement!.type} #${highlightedElement!.index}`);
        element!.click();
      }, 1500);
    } else {
      console.warn(`Could not find element with selector: ${highlightedElement.selector}`);
      // Try again with a different link/button
      highlightedElement = null;
      for (let i = 0; i < Math.min(5, linksAndButtons.length); i++) {
        const tryIndex = Math.floor(Math.random() * linksAndButtons.length);
        const tryElement = getElementBySelector(linksAndButtons[tryIndex].selector);
        if (tryElement) {
          highlightedElement = linksAndButtons[tryIndex];
          console.log(`Highlighting ${highlightedElement.type} #${highlightedElement.index}: ${highlightedElement.text}`);
          highlightElement(tryElement);
          movePointerToElement(tryElement);
          
          // Click the element after a short delay
          setTimeout(() => {
            console.log(`Clicking ${highlightedElement!.type} #${highlightedElement!.index}`);
            tryElement.click();
          }, 1500);
          break;
        }
      }
    }
  }

  const message = `
📄 **Page**: ${pageTitle}
🔗 **URL**: ${pageUrl}

**Found ${uniqueElements.length} interactive elements**
- ${uniqueElements.filter(e => e.type === 'link').length} links
- ${uniqueElements.filter(e => e.type === 'button').length} buttons
- ${uniqueElements.filter(e => e.type === 'input').length} inputs

${highlightedElement ? `\n**🎯 Highlighted #${highlightedElement.index}:** ${highlightedElement.type} - "${highlightedElement.text}"` : ''}

**Your Prompt**: "${prompt}"

---

**Interactive Elements (JSON):**
\`\`\`json
${JSON.stringify(uniqueElements, null, 2)}
\`\`\`
  `.trim();

  console.log('Interactive Elements:', JSON.stringify(uniqueElements, null, 2));

  return {
    message,
    elements: uniqueElements
  };
}

/**
 * Get group information for an element based on its parent containers
 */
function getGroupInfo(element: Element): { id: string; label: string } {
  let current: Element | null = element.parentElement;
  let depth = 0;
  const maxDepth = 5;

  // Look for semantic parent containers
  while (current && depth < maxDepth) {
    const tag = current.tagName.toLowerCase();
    
    // Check for semantic HTML5 elements
    if (['article', 'section', 'aside', 'nav', 'form'].includes(tag)) {
      const label = current.getAttribute('aria-label') || 
                    current.getAttribute('data-testid') ||
                    current.className.split(' ')[0] || 
                    tag;
      return {
        id: generateGroupId(current),
        label: label.substring(0, 50)
      };
    }
    
    // Check for common container patterns
    const classes = current.className.toLowerCase();
    if (classes.includes('card') || classes.includes('product') || 
        classes.includes('item') || classes.includes('container') ||
        classes.includes('post') || classes.includes('listing')) {
      const label = current.getAttribute('aria-label') || 
                    current.className.split(' ')[0] ||
                    'container';
      return {
        id: generateGroupId(current),
        label: label.substring(0, 50)
      };
    }
    
    current = current.parentElement;
    depth++;
  }
  
  // Fallback to body if no semantic parent found
  return {
    id: 'root',
    label: 'page'
  };
}

/**
 * Generate a unique ID for a group container
 */
function generateGroupId(element: Element): string {
  if (element.id) {
    return `group-${element.id}`;
  }
  
  const tag = element.tagName.toLowerCase();
  const classes = element.className.split(' ')[0] || '';
  const siblings = element.parentElement?.children;
  
  if (siblings) {
    const index = Array.from(siblings).indexOf(element);
    return `group-${tag}-${classes}-${index}`.replace(/\s+/g, '-').toLowerCase();
  }
  
  return `group-${tag}-${classes}`.replace(/\s+/g, '-').toLowerCase();
}

/**
 * Generate a unique CSS selector for an element
 * Stores a reference to the actual element for later retrieval
 */
const elementCache = new Map<string, HTMLElement>();
let selectorCounter = 0;

/**
 * Get element by selector, using cache for cached selectors
 */
function getElementBySelector(selector: string): HTMLElement | null {
  // Check if this is a cached selector
  const cacheMatch = selector.match(/\[data-cached-selector="([^"]+)"\]/);
  if (cacheMatch) {
    const cachedId = cacheMatch[1];
    const cachedElement = elementCache.get(cachedId);
    if (cachedElement && document.body.contains(cachedElement)) {
      return cachedElement;
    }
  }
  
  // Fall back to regular querySelector
  try {
    return document.querySelector(selector) as HTMLElement;
  } catch (e) {
    console.warn('Invalid selector:', selector, e);
    return null;
  }
}

function generateSelector(element: Element): string {
  // Use ID if available and valid
  if (element.id) {
    // Check if ID contains special characters that need escaping (only allow alphanumeric, dash, underscore)
    if (/^[a-zA-Z][\w-]*$/.test(element.id)) {
      const selector = `#${element.id}`;
      try {
        // Verify it's unique
        if (document.querySelectorAll(selector).length === 1) {
          return selector;
        }
      } catch (e) {
        // Invalid selector even after validation, skip it
      }
    }
  }
  
  // Try href for links (most stable identifier)
  if (element.tagName === 'A') {
    const href = (element as HTMLAnchorElement).href;
    if (href) {
      const linkSelector = `a[href="${href}"]`;
      try {
        const matches = document.querySelectorAll(linkSelector);
        if (matches.length === 1) {
          return linkSelector;
        } else if (matches.length > 1) {
          // Multiple links with same href, cache the element
          const uniqueId = `cached-element-${selectorCounter++}`;
          elementCache.set(uniqueId, element as HTMLElement);
          (element as any).__cachedSelector = uniqueId;
          element.setAttribute('data-cached-selector', uniqueId);
          return `[data-cached-selector="${uniqueId}"]`;
        }
      } catch (e) {
        // Invalid selector, fall through
      }
    }
  }
  
  // Try name attribute for inputs
  if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
    const name = element.getAttribute('name');
    if (name) {
      const nameSelector = `${element.tagName.toLowerCase()}[name="${name}"]`;
      try {
        if (document.querySelectorAll(nameSelector).length === 1) {
          return nameSelector;
        }
      } catch (e) {
        // Invalid selector, fall through
      }
    }
  }
  
  // Try unique class combinations
  const classes = Array.from(element.classList).filter(c => c && !c.match(/^(hover|focus|active|selected|disabled)/));
  if (classes.length > 0) {
    const selector = `${element.tagName.toLowerCase()}.${classes.join('.')}`;
    try {
      if (document.querySelectorAll(selector).length === 1) {
        return selector;
      }
    } catch (e) {
      // Invalid selector, fall through
    }
  }
  
  // Cache the element with a unique identifier as last resort
  const uniqueId = `cached-element-${selectorCounter++}`;
  elementCache.set(uniqueId, element as HTMLElement);
  (element as any).__cachedSelector = uniqueId;
  
  // Mark the element with a data attribute for future reference
  element.setAttribute('data-cached-selector', uniqueId);
  
  return `[data-cached-selector="${uniqueId}"]`;
}

/**
 * Move a visual pointer to an element and scroll it into view
 * Note: JavaScript cannot control the actual system cursor for security reasons,
 * so this creates a prominent visual indicator instead
 */
function movePointerToElement(element: HTMLElement): void {
  // Scroll element into view with smooth behavior first
  element.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
  
  // Wait a bit for scroll to complete, then position pointer
  setTimeout(() => {
    // Create or update custom pointer (cursor icon)
    let pointer = document.getElementById('big-brother-pointer');
    if (!pointer) {
      pointer = document.createElement('div');
      pointer.id = 'big-brother-pointer';
      pointer.innerHTML = '👆'; // Pointing hand emoji
      pointer.style.position = 'fixed';
      pointer.style.fontSize = '48px';
      pointer.style.pointerEvents = 'none';
      pointer.style.zIndex = '999999';
      pointer.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
      pointer.style.filter = 'drop-shadow(0 0 10px rgba(255, 255, 255, 0.9)) drop-shadow(0 0 20px rgba(255, 0, 0, 0.8))';
      document.body.appendChild(pointer);
    }
    
    // Get element position after scroll
    const rect = element.getBoundingClientRect();
    
    // Check if element is actually visible
    if (rect.width === 0 || rect.height === 0) {
      console.warn('Element has zero dimensions, skipping pointer');
      return;
    }
    
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Verify position is reasonable (not off-screen)
    if (centerX < -100 || centerX > window.innerWidth + 100 || centerY < -100 || centerY > window.innerHeight + 100) {
      console.warn('Element position is off-screen, adjusting');
    }
    
    // Position pointer above element
    pointer.style.left = `${centerX - 24}px`;
    pointer.style.top = `${centerY - 60}px`;
    pointer.style.opacity = '1';
    pointer.style.transform = 'scale(1) translateY(0)';
    
    // Bounce animation
    let bounceCount = 0;
    const bounceInterval = setInterval(() => {
      if (bounceCount % 2 === 0) {
        pointer!.style.transform = 'scale(1.2) translateY(-10px)';
      } else {
        pointer!.style.transform = 'scale(1) translateY(0)';
      }
      bounceCount++;
      if (bounceCount >= 6) {
        clearInterval(bounceInterval);
      }
    }, 300);
    
    // Fade out after 4 seconds
    setTimeout(() => {
      pointer!.style.opacity = '0';
      pointer!.style.transform = 'scale(0.5) translateY(-20px)';
    }, 4000);
  }, 300); // Wait 300ms for scroll to start
}

/**
 * Highlight an element visually
 */
function highlightElement(element: HTMLElement): void {
  const computedStyle = window.getComputedStyle(element);
  const bgColor = computedStyle.backgroundColor;
  
  let highlightColor = '#FF00FF';
  const rgbMatch = bgColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (rgbMatch) {
    const r = 255 - parseInt(rgbMatch[1]);
    const g = 255 - parseInt(rgbMatch[2]);
    const b = 255 - parseInt(rgbMatch[3]);
    highlightColor = `rgb(${r}, ${g}, ${b})`;
  }
  
  element.style.backgroundColor = highlightColor;
  element.style.color = rgbMatch && (parseInt(rgbMatch[1]) > 128) ? '#000000' : '#FFFFFF';
  element.style.border = '3px solid #FF0000';
  element.style.padding = '8px';
  element.style.fontWeight = 'bold';
  element.style.boxShadow = '0 0 20px rgba(255, 0, 0, 0.8)';
  element.style.transform = 'scale(1.1)';
  element.style.transition = 'all 0.3s ease';
  element.style.zIndex = '9999';
  element.style.position = 'relative';
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
