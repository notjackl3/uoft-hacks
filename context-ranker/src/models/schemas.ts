/**
 * MongoDB Collection Schemas for Context Ranker
 */

import { ObjectId } from 'mongodb';

// ============================================
// PAGES COLLECTION
// Stores crawled page metadata
// ============================================
export interface Page {
  _id?: ObjectId;
  url: string;
  domain: string;
  path: string;
  title: string;
  headings: string[];
  crawledAt: Date;
  lastModified?: Date;
  contentHash: string; // For change detection
  statusCode: number;
  passageCount: number;
}

// ============================================
// PASSAGES COLLECTION
// Stores chunked content with embeddings and features
// ============================================
export interface PassageFeatures {
  hasSteps: boolean;         // Contains numbered/bullet steps
  hasCode: boolean;          // Contains code blocks
  hasUIWords: boolean;       // Contains UI-related words (click, button, menu, etc.)
  marketingScore: number;    // 0-1, higher = more marketing fluff
  authorityTier: number;     // 1-3, based on path (/docs=1, /blog=3)
  freshnessDays: number;     // Days since page was last modified
  chunkPosition: number;     // Position in original document (0 = first)
  totalChunks: number;       // Total chunks from this page
  tokenCount: number;        // Actual token count of passage
}

export interface Passage {
  _id?: ObjectId;
  pageId: ObjectId;
  url: string;
  domain: string;
  title: string;
  content: string;
  embedding: number[];       // Vector embedding
  features: PassageFeatures;
  createdAt: Date;
  updatedAt: Date;
}

// ============================================
// TRAINING_EVENTS COLLECTION
// Stores user feedback for XGBoost training
// ============================================
export interface TrainingEvent {
  _id?: ObjectId;
  sessionId: string;
  task: string;
  outcome: 'success' | 'failure' | 'partial' | 'abandoned';
  
  // Chosen and candidate passages
  chosenPassageId?: string;
  candidatePassageIds: string[];
  
  // Features for training
  chosenFeatures?: PassageFeatures | null;
  candidateFeatures?: { passageId: string; features: PassageFeatures }[];
  
  // Additional metadata
  userFeedback?: string;
  actionsTaken?: number;
  timeSpentMs?: number;
  
  createdAt: Date;
}

// ============================================
// API Request/Response Types
// ============================================
export interface ContextRequest {
  site: string;              // Domain to search (e.g., "stripe.com")
  task: string;              // What the user is trying to do
  currentUrl: string;        // Where the user currently is
  visibleTextHints?: string; // Optional visible text on page
}

export interface RankedPassage {
  id: string;
  url: string;
  title: string;
  content: string;
  score: number;
  features: PassageFeatures;
}

export interface ContextResponse {
  passages: RankedPassage[];
  queryId: string;           // For feedback tracking
  searchTimeMs: number;
}

export interface FeedbackRequest {
  queryId: string;
  selectedPassageId?: string;
  wasHelpful: boolean;
  notes?: string;
}

// ============================================
// Ingestion Types
// ============================================
export interface ExtractedContent {
  title: string;
  headings: string[];
  mainText: string;
  codeBlocks: string[];
}

export interface CrawlResult {
  url: string;
  statusCode: number;
  html?: string;
  error?: string;
}
