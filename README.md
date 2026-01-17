# Big Brother - AI Agent Chrome Extension

A Manifest V3 Chrome Extension that acts as an AI Agent interface, allowing users to interact with webpages through natural language prompts.

## 🚀 Tech Stack

- **React 18** with **TypeScript** (Strict mode)
- **Vite** as the build tool
- **TailwindCSS** for styling
- **CRXJS** for Chrome Extension development
- **Manifest V3** architecture

## 📁 Project Structure

```
big-brother-extension/
├── src/
│   ├── background/
│   │   └── background.ts          # Service Worker for message routing
│   ├── content/
│   │   └── content.ts              # Content script injected into pages
│   ├── sidepanel/
│   │   ├── App.tsx                 # Root component
│   │   ├── SidePanel.tsx           # Main Side Panel UI
│   │   ├── index.tsx               # React entry point
│   │   └── index.css               # TailwindCSS styles
│   ├── types/
│   │   └── messages.ts             # TypeScript type definitions
│   └── utils/
│       └── messaging.ts            # Message passing utilities
├── manifest.json                   # Chrome Extension manifest
├── vite.config.ts                  # Vite configuration
├── tsconfig.json                   # TypeScript configuration
├── tailwind.config.js              # TailwindCSS configuration
├── postcss.config.js               # PostCSS configuration
├── package.json                    # Dependencies and scripts
└── index.html                      # Side Panel HTML entry
```

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Build the Extension

For development (with hot reload):
```bash
npm run dev
```

For production build:
```bash
npm run build
```

### 3. Load Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right corner)
3. Click **Load unpacked**
4. Select the `dist` folder from your project directory

### 4. Using the Extension

1. Click the Big Brother extension icon in your Chrome toolbar
2. The Side Panel will open on the right side of the browser
3. Navigate to any webpage
4. Type a prompt in the Side Panel (e.g., "Change my username" or "Inspect this page")
5. Press Enter or click Send
6. The agent will inspect the DOM and respond

## 🎯 Core Features

### Side Panel
- **Chat Interface**: Scrollable chat history with user and agent messages
- **Status Indicator**: Visual feedback (Idle, Thinking, Acting)
- **Persistent Storage**: Chat history saved using Chrome Storage API
- **Clean UI**: Built with TailwindCSS for a modern look

### Background Service Worker
- Routes messages between Side Panel and Content Script
- Handles extension lifecycle events
- Opens Side Panel when extension icon is clicked

### Content Script
- Injected into all pages
- Full DOM access for inspection and interaction
- Helper functions for common DOM operations:
  - `queryDOMElements()` - Query DOM elements
  - `clickElement()` - Click elements
  - `fillInput()` - Fill input fields

### Message Passing
- Type-safe messaging system with TypeScript
- Robust error handling
- Async/await support

## 🔌 Ready for LLM Integration

The extension is scaffolded and ready for you to integrate your LLM API:

1. **In SidePanel.tsx** (line ~67): Replace the simulated response with your LLM API call
2. **In content.ts** (line ~36): Enhance `handleUserPrompt()` to parse LLM instructions
3. **Add new helper functions** in content.ts for specific DOM actions

Example integration point:
```typescript
// In SidePanel.tsx
const response = await sendMessageToContent({
  type: 'USER_PROMPT',
  payload: { prompt: inputValue },
});

// TODO: Call your LLM API here
// const llmResponse = await callLLMAPI(inputValue, response);
```

## 📝 Development Commands

- `npm run dev` - Start development server with HMR
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run type-check` - Run TypeScript type checking

## 🔐 Permissions

The extension requests the following permissions:
- `activeTab` - Access to the current active tab
- `scripting` - Ability to inject scripts
- `storage` - Store chat history
- `sidePanel` - Display the side panel UI
- `<all_urls>` - Access to all websites

## 🎨 Customization

### Styling
Modify [tailwind.config.js](tailwind.config.js) to customize the theme.

### Icons
Add your own icons to the `icons/` folder and update [manifest.json](manifest.json):
- `icon16.png` - 16x16px
- `icon48.png` - 48x48px
- `icon128.png` - 128x128px

### Content Script Matching
Update `manifest.json` to change which pages the content script runs on:
```json
"content_scripts": [{
  "matches": ["<all_urls>"],  // Change to specific domains
  "js": ["src/content/content.ts"]
}]
```

## 🐛 Debugging

### View Extension Logs
1. **Background Service Worker**: 
   - Go to `chrome://extensions/`
   - Click "Service worker" under Big Brother

2. **Content Script**: 
   - Open DevTools on any webpage (F12)
   - Check Console tab

3. **Side Panel**: 
   - Right-click on Side Panel
   - Select "Inspect"

### Common Issues

**Side Panel won't open**: Make sure you have an active tab open.

**Content script not responding**: Check that the content script is injected by looking at the page's console logs.

**Build errors**: Try deleting `node_modules` and `dist` folders, then run `npm install` again.

## 📚 Next Steps

1. **Integrate an LLM API** (OpenAI, Anthropic, etc.)
2. **Add DOM action parsing** to execute user commands
3. **Implement visual feedback** for DOM interactions
4. **Add authentication** if using external APIs
5. **Create preset prompts** for common tasks
6. **Add error boundaries** for better error handling

## 📄 License

MIT License - Feel free to use this as a starting point for your own extensions!

---

Built with ❤️ using React, TypeScript, and Vite
