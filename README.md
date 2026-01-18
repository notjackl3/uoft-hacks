# Big Brother - AI Agent Chrome Extension

A Manifest V3 Chrome Extension that acts as an AI Agent interface, allowing users to interact with webpages through natural language prompts.

**Landing Page:** [GitHub](https://github.com/phintruong/Big-Bro-UoftHack) | [Live Site](https://big-bro-please-help-me-with-this-new.tech/)

## :rocket: Tech Stack

**Frontend (Extension):**
- **React 18** with **TypeScript** (Strict mode) - Component-based UI library
- **Vite** - Fast build tool and dev server with HMR
- **TailwindCSS** - Utility-first CSS framework for styling
- **CRXJS** - Vite plugin for Chrome Extension development
- **Manifest V3** - Modern Chrome Extension architecture

**Backend (API):**
- **FastAPI** - High-performance Python web framework
- **MongoDB** (Motor) - Async database for session tracking
- **OpenAI API** - LLM integration for natural language processing
- **Voyage AI** - Embeddings for semantic matching
- **Backboard SDK** - Unified AI API interface

**Landing Page:**
- **React** + **TypeScript** - Frontend framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn-ui** - UI component library
- **Radix UI** - Accessible primitives

## :wrench: Setup

### Install Dependencies
```bash
npm install
```

### Build the Extension
```bash
# Development (with hot reload)
npm run dev

# Production
npm run build
```

### Load Extension in Chrome
1. Open `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `dist` folder

## :dart: Core Features

- **Side Panel Interface** - Chat-based UI for natural language interaction
- **AI Agent Integration** - Connect to backend API for workflow planning
- **DOM Inspection** - Analyze and interact with webpage elements
- **Visual Highlighting** - Step-by-step guidance with element highlighting
- **Session Tracking** - Persistent session management via MongoDB
- **Speech-to-Text** - Voice input support using Web Speech API
- **Text-to-Speech** - Voice output via ElevenLabs API

## :pencil: Development Commands

- `npm run dev` - Start development server with HMR
- `npm run build` - Build for production
- `npm run type-check` - Run TypeScript type checking
