/**
 * Text Chunker
 * 
 * Splits text into chunks of 300-450 tokens with ~50 token overlap.
 * Tries to break at sentence/paragraph boundaries when possible.
 */

import { encode, decode } from 'gpt-tokenizer';

export interface ChunkOptions {
  minTokens?: number;
  maxTokens?: number;
  overlapTokens?: number;
}

export interface Chunk {
  text: string;
  tokenCount: number;
  position: number;      // 0-indexed position in document
  startChar: number;     // Character offset in original text
  endChar: number;
}

const DEFAULT_OPTIONS: Required<ChunkOptions> = {
  minTokens: 300,
  maxTokens: 450,
  overlapTokens: 50,
};

export function chunkText(text: string, options: ChunkOptions = {}): Chunk[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const chunks: Chunk[] = [];

  if (!text.trim()) {
    return chunks;
  }

  // Tokenize the entire text
  const tokens = encode(text);
  
  if (tokens.length <= opts.maxTokens) {
    // Text fits in a single chunk
    return [{
      text: text.trim(),
      tokenCount: tokens.length,
      position: 0,
      startChar: 0,
      endChar: text.length,
    }];
  }

  // Split into sentences for better chunking
  const sentences = splitIntoSentences(text);
  
  let currentChunk: string[] = [];
  let currentTokens = 0;
  let chunkStartChar = 0;
  let charOffset = 0;
  let position = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    const sentenceTokens = encode(sentence).length;

    // If single sentence exceeds max, we need to split it
    if (sentenceTokens > opts.maxTokens) {
      // Flush current chunk if not empty
      if (currentChunk.length > 0) {
        const chunkText = currentChunk.join(' ').trim();
        chunks.push({
          text: chunkText,
          tokenCount: currentTokens,
          position,
          startChar: chunkStartChar,
          endChar: charOffset,
        });
        position++;
        currentChunk = [];
        currentTokens = 0;
      }

      // Split long sentence by tokens
      const splitChunks = splitLongText(sentence, opts.maxTokens, opts.overlapTokens);
      for (const sc of splitChunks) {
        chunks.push({
          text: sc.text,
          tokenCount: sc.tokenCount,
          position,
          startChar: charOffset,
          endChar: charOffset + sentence.length,
        });
        position++;
      }
      
      charOffset += sentence.length + 1;
      chunkStartChar = charOffset;
      continue;
    }

    // Check if adding this sentence would exceed max
    if (currentTokens + sentenceTokens > opts.maxTokens) {
      // Flush current chunk
      if (currentChunk.length > 0) {
        const chunkText = currentChunk.join(' ').trim();
        chunks.push({
          text: chunkText,
          tokenCount: currentTokens,
          position,
          startChar: chunkStartChar,
          endChar: charOffset,
        });
        position++;

        // Keep overlap sentences
        const { overlapSentences, overlapTokens } = getOverlap(
          currentChunk,
          opts.overlapTokens
        );
        currentChunk = overlapSentences;
        currentTokens = overlapTokens;
        chunkStartChar = charOffset - overlapSentences.join(' ').length;
      }
    }

    currentChunk.push(sentence);
    currentTokens += sentenceTokens;
    charOffset += sentence.length + 1; // +1 for space/newline
  }

  // Flush remaining
  if (currentChunk.length > 0 && currentTokens >= opts.minTokens / 2) {
    const chunkText = currentChunk.join(' ').trim();
    chunks.push({
      text: chunkText,
      tokenCount: currentTokens,
      position,
      startChar: chunkStartChar,
      endChar: charOffset,
    });
  }

  return chunks;
}

function splitIntoSentences(text: string): string[] {
  // Split on sentence boundaries, keeping the delimiter
  const sentences: string[] = [];
  
  // First split on paragraphs
  const paragraphs = text.split(/\n\n+/);
  
  for (const para of paragraphs) {
    // Then split on sentences
    const sentenceRegex = /[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$/g;
    const matches = para.match(sentenceRegex) || [para];
    
    for (const match of matches) {
      const trimmed = match.trim();
      if (trimmed) {
        sentences.push(trimmed);
      }
    }
  }
  
  return sentences;
}

function splitLongText(
  text: string,
  maxTokens: number,
  overlapTokens: number
): { text: string; tokenCount: number }[] {
  const tokens = encode(text);
  const chunks: { text: string; tokenCount: number }[] = [];
  
  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(start + maxTokens, tokens.length);
    const chunkTokens = tokens.slice(start, end);
    const chunkText = decode(chunkTokens);
    
    chunks.push({
      text: chunkText.trim(),
      tokenCount: chunkTokens.length,
    });
    
    start = end - overlapTokens;
    if (start >= tokens.length - overlapTokens) {
      break;
    }
  }
  
  return chunks;
}

function getOverlap(
  sentences: string[],
  targetOverlapTokens: number
): { overlapSentences: string[]; overlapTokens: number } {
  const overlapSentences: string[] = [];
  let overlapTokens = 0;
  
  // Work backwards from the end
  for (let i = sentences.length - 1; i >= 0; i--) {
    const sentenceTokens = encode(sentences[i]).length;
    if (overlapTokens + sentenceTokens <= targetOverlapTokens * 1.5) {
      overlapSentences.unshift(sentences[i]);
      overlapTokens += sentenceTokens;
    } else {
      break;
    }
  }
  
  return { overlapSentences, overlapTokens };
}

// Utility to count tokens
export function countTokens(text: string): number {
  return encode(text).length;
}
