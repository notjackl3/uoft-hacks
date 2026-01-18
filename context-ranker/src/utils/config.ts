/**
 * Configuration loader from environment variables
 */

import dotenv from 'dotenv';
import path from 'path';

// Load .env file
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

export const config = {
  // MongoDB
  mongodbUri: process.env.MONGODB_URI || 'mongodb://localhost:27017/context-ranker',
  
  // Embedding Provider: 'backboard' or 'openai'
  embeddingProvider: process.env.EMBEDDING_PROVIDER || 'backboard',
  embeddingApiKey: process.env.EMBEDDING_API_KEY || '',
  embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
  backboardBaseUrl: process.env.BACKBOARD_BASE_URL || 'https://api.backboard.io',
  
  // Server
  port: parseInt(process.env.PORT || '3001', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  
  // XGBoost
  useXgboost: process.env.USE_XGBOOST === 'true',
  
  // Crawler
  maxPagesPerDomain: parseInt(process.env.MAX_PAGES_PER_DOMAIN || '100', 10),
  crawlDelayMs: parseInt(process.env.CRAWL_DELAY_MS || '1000', 10),
  
  // Retrieval
  candidateLimit: parseInt(process.env.CANDIDATE_LIMIT || '500', 10),
  topKResults: parseInt(process.env.TOP_K_RESULTS || '10', 10),
};

// Validate required config
export function validateConfig(): void {
  const missing: string[] = [];
  
  if (!config.embeddingApiKey) {
    missing.push('EMBEDDING_API_KEY');
  }
  
  if (missing.length > 0) {
    console.warn(`⚠️  Missing environment variables: ${missing.join(', ')}`);
    console.warn('Some features may not work correctly.');
  }
}
