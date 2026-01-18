/**
 * Embedding Provider Interface
 * 
 * Supports multiple embedding providers:
 * - backboard: Uses Backboard.io embedding API
 * - openai: Uses OpenAI embeddings (fallback)
 */

import { config } from '../utils/config';

export interface EmbeddingProvider {
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

// ============================================
// BACKBOARD PROVIDER
// Uses Backboard.io's embedding endpoint
// ============================================
class BackboardEmbeddingProvider implements EmbeddingProvider {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl: string = 'https://api.backboard.io') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: text,
        model: config.embeddingModel,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Backboard embedding failed: ${response.status} - ${error}`);
    }

    const data = await response.json() as any;
    return data.data?.[0]?.embedding || data.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: texts,
        model: config.embeddingModel,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Backboard batch embedding failed: ${response.status} - ${error}`);
    }

    const data = await response.json() as any;
    
    // Handle both array and single response formats
    if (Array.isArray(data.data)) {
      return data.data.map((item: any) => item.embedding);
    }
    return [data.embedding];
  }
}

// ============================================
// OPENAI PROVIDER (Fallback)
// ============================================
class OpenAIEmbeddingProvider implements EmbeddingProvider {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: text,
        model: config.embeddingModel || 'text-embedding-3-small',
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI embedding failed: ${response.status} - ${error}`);
    }

    const data = await response.json() as any;
    return data.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: texts,
        model: config.embeddingModel || 'text-embedding-3-small',
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI batch embedding failed: ${response.status} - ${error}`);
    }

    const data = await response.json() as any;
    return data.data.map((item: any) => item.embedding);
  }
}

// ============================================
// FACTORY
// ============================================
let embeddingProvider: EmbeddingProvider | null = null;

export function getEmbeddingProvider(): EmbeddingProvider {
  if (embeddingProvider) return embeddingProvider;

  const provider = config.embeddingProvider;
  const apiKey = config.embeddingApiKey;

  if (!apiKey) {
    throw new Error('EMBEDDING_API_KEY is required');
  }

  switch (provider) {
    case 'backboard':
      console.log('🔌 Using Backboard embedding provider');
      embeddingProvider = new BackboardEmbeddingProvider(apiKey, config.backboardBaseUrl);
      break;
    case 'openai':
    default:
      console.log('🔌 Using OpenAI embedding provider');
      embeddingProvider = new OpenAIEmbeddingProvider(apiKey);
      break;
  }

  return embeddingProvider;
}

// Convenience functions
export async function embedText(text: string): Promise<number[]> {
  return getEmbeddingProvider().embed(text);
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  return getEmbeddingProvider().embedBatch(texts);
}
