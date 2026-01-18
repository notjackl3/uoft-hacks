/**
 * POST /context API Endpoint
 * 
 * Takes: { site, task, currentUrl, visibleTextHints? }
 * Returns: top-ranked passages with URLs
 */

import { Router, Request, Response } from 'express';
import { ObjectId } from 'mongodb';
import { getPassagesCollection } from '../services/database';
import { embedText } from '../services/embeddings';
import { Passage } from '../models/schemas';
import { config } from '../utils/config';
import { getXGBoostScores, isXGBoostAvailable } from '../xgboost/inference';

const router = Router();

interface ContextRequest {
  site: string;          // Domain to search (e.g., "stripe.com")
  task: string;          // User's task/question
  currentUrl?: string;   // Current page URL for context
  visibleTextHints?: string;  // Visible text on current page
}

interface RankedPassage {
  url: string;
  title: string;
  content: string;
  score: number;
  features: {
    hasSteps: boolean;
    hasCode: boolean;
    hasUIWords: boolean;
    authorityTier: number;
  };
}

interface ContextResponse {
  passages: RankedPassage[];
  query: string;
  candidatesScanned: number;
  timeTakenMs: number;
}

// Cosine similarity between two vectors
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}

// Heuristic scoring weights
const WEIGHTS = {
  semanticSimilarity: 0.50,   // Cosine similarity of embeddings
  hasSteps: 0.10,             // Bonus for step-by-step content
  hasCode: 0.08,              // Bonus for code examples
  hasUIWords: 0.12,           // Bonus for UI-related content (good for browser automation)
  authorityTier: 0.10,        // Bonus for authoritative sources (tier 1 = docs)
  marketingPenalty: 0.05,     // Penalty for marketing content
  freshnessBonus: 0.05,       // Bonus for recent content
};

// Calculate heuristic score for a passage
function calculateScore(
  passage: Passage,
  queryEmbedding: number[],
  taskLower: string
): number {
  // Base: semantic similarity
  const semanticScore = passage.embedding 
    ? cosineSimilarity(queryEmbedding, passage.embedding) 
    : 0;
  
  let score = semanticScore * WEIGHTS.semanticSimilarity;
  
  // Feature bonuses
  if (passage.features.hasSteps) {
    score += WEIGHTS.hasSteps;
  }
  
  if (passage.features.hasCode) {
    score += WEIGHTS.hasCode;
  }
  
  if (passage.features.hasUIWords) {
    score += WEIGHTS.hasUIWords;
  }
  
  // Authority tier (1 = best, 3 = worst)
  const authorityBonus = (4 - passage.features.authorityTier) / 3 * WEIGHTS.authorityTier;
  score += authorityBonus;
  
  // Marketing penalty
  score -= passage.features.marketingScore * WEIGHTS.marketingPenalty;
  
  // Freshness bonus (decay over 30 days)
  const freshnessFactor = Math.max(0, 1 - passage.features.freshnessDays / 30);
  score += freshnessFactor * WEIGHTS.freshnessBonus;
  
  // Task-specific bonuses
  if (taskLower.includes('how to') || taskLower.includes('step')) {
    if (passage.features.hasSteps) score += 0.05;
  }
  
  if (taskLower.includes('code') || taskLower.includes('example') || taskLower.includes('api')) {
    if (passage.features.hasCode) score += 0.05;
  }
  
  if (taskLower.includes('click') || taskLower.includes('button') || taskLower.includes('navigate')) {
    if (passage.features.hasUIWords) score += 0.05;
  }
  
  return score;
}

// Build query string from request
function buildQueryString(req: ContextRequest): string {
  const parts: string[] = [req.task];
  
  if (req.visibleTextHints) {
    // Take first 200 chars of visible text
    parts.push(req.visibleTextHints.substring(0, 200));
  }
  
  if (req.currentUrl) {
    // Extract path tokens from URL
    try {
      const url = new URL(req.currentUrl);
      const pathTokens = url.pathname
        .split(/[\/\-_]/)
        .filter(t => t.length > 2)
        .join(' ');
      if (pathTokens) parts.push(pathTokens);
    } catch {
      // Invalid URL, skip
    }
  }
  
  return parts.join(' ');
}

router.post('/', async (req: Request, res: Response) => {
  const startTime = Date.now();
  
  try {
    const body = req.body as ContextRequest;
    
    // Validate request
    if (!body.site || !body.task) {
      return res.status(400).json({
        error: 'Missing required fields: site, task'
      });
    }
    
    const passagesCol = getPassagesCollection();
    const queryString = buildQueryString(body);
    const taskLower = body.task.toLowerCase();
    
    console.log(`🔍 Context query: "${body.task}" for site: ${body.site}`);
    
    // Step 1: Get candidate passages via text search
    // If MongoDB Atlas text search isn't available, we fall back to domain filtering
    let candidates: Passage[];
    
    // Extract domain from site (handle both "stripe.com" and "https://stripe.com")
    let domain = body.site;
    try {
      if (body.site.includes('://')) {
        domain = new URL(body.site).hostname;
      }
    } catch {
      // Keep original
    }
    
    // Try text search first, fall back to domain-only query
    try {
      candidates = await passagesCol.find({
        domain: { $regex: domain, $options: 'i' },
        $text: { $search: queryString }
      }, {
        projection: { score: { $meta: 'textScore' } },
        limit: 500
      }).sort({ score: { $meta: 'textScore' } }).toArray() as Passage[];
    } catch {
      // Text search not available, use domain filter only
      candidates = await passagesCol.find({
        domain: { $regex: domain, $options: 'i' }
      }).limit(500).toArray() as Passage[];
    }
    
    console.log(`   📦 Found ${candidates.length} candidates from ${domain}`);
    
    if (candidates.length === 0) {
      return res.json({
        passages: [],
        query: queryString,
        candidatesScanned: 0,
        timeTakenMs: Date.now() - startTime
      } as ContextResponse);
    }
    
    // Step 2: Compute query embedding
    console.log(`   🔢 Computing query embedding...`);
    const queryEmbedding = await embedText(queryString);
    
    // Step 3: Score all candidates
    let scoredPassages: { passage: Passage; score: number }[];
    
    // Use XGBoost if enabled and available
    if (config.useXgboost && isXGBoostAvailable()) {
      console.log(`   🌲 Using XGBoost reranker...`);
      const features = candidates.map(p => p.features);
      const xgbScores = await getXGBoostScores(features);
      
      if (xgbScores.length === candidates.length) {
        // Combine XGBoost scores with semantic similarity
        scoredPassages = candidates.map((passage, i) => {
          const semanticScore = passage.embedding 
            ? cosineSimilarity(queryEmbedding, passage.embedding) 
            : 0;
          // Weighted combination: 60% XGBoost, 40% semantic
          const combinedScore = 0.6 * xgbScores[i] + 0.4 * semanticScore;
          return { passage, score: combinedScore };
        });
      } else {
        // Fallback to heuristic scoring
        scoredPassages = candidates.map(passage => ({
          passage,
          score: calculateScore(passage, queryEmbedding, taskLower)
        }));
      }
    } else {
      // Heuristic scoring
      scoredPassages = candidates.map(passage => ({
        passage,
        score: calculateScore(passage, queryEmbedding, taskLower)
      }));
    }
    
    // Step 4: Sort by score and take top results
    scoredPassages.sort((a, b) => b.score - a.score);
    const topPassages = scoredPassages.slice(0, 10);
    
    // Step 5: Format response
    const response: ContextResponse = {
      passages: topPassages.map(({ passage, score }) => ({
        url: passage.url,
        title: passage.title,
        content: passage.content,
        score: Math.round(score * 1000) / 1000,
        features: {
          hasSteps: passage.features.hasSteps,
          hasCode: passage.features.hasCode,
          hasUIWords: passage.features.hasUIWords,
          authorityTier: passage.features.authorityTier,
        }
      })),
      query: queryString,
      candidatesScanned: candidates.length,
      timeTakenMs: Date.now() - startTime
    };
    
    console.log(`   ✅ Returning ${response.passages.length} passages (${response.timeTakenMs}ms)`);
    
    return res.json(response);
    
  } catch (error: any) {
    console.error('❌ Context query error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

export default router;
