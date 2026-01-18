/**
 * POST /feedback API Endpoint
 * 
 * Logs outcomes and chosen passage IDs for training data collection.
 */

import { Router, Request, Response } from 'express';
import { ObjectId } from 'mongodb';
import { getTrainingEventsCollection, getPassagesCollection } from '../services/database';
import { TrainingEvent } from '../models/schemas';

const router = Router();

interface FeedbackRequest {
  sessionId: string;           // Unique session identifier
  task: string;                // Original task
  chosenPassageId?: string;    // ID of passage user found helpful
  outcome: 'success' | 'failure' | 'partial' | 'abandoned';
  candidatePassageIds?: string[];  // All passages that were shown
  userFeedback?: string;       // Optional user comment
  actionsTaken?: number;       // Number of actions taken
  timeSpentMs?: number;        // Time spent on task
}

router.post('/', async (req: Request, res: Response) => {
  try {
    const body = req.body as FeedbackRequest;
    
    // Validate request
    if (!body.sessionId || !body.task || !body.outcome) {
      return res.status(400).json({
        error: 'Missing required fields: sessionId, task, outcome'
      });
    }
    
    const validOutcomes = ['success', 'failure', 'partial', 'abandoned'];
    if (!validOutcomes.includes(body.outcome)) {
      return res.status(400).json({
        error: `Invalid outcome. Must be one of: ${validOutcomes.join(', ')}`
      });
    }
    
    const trainingCol = getTrainingEventsCollection();
    const passagesCol = getPassagesCollection();
    
    // Get feature vectors for chosen and candidate passages
    let chosenFeatures = null;
    let candidateFeatures: any[] = [];
    
    if (body.chosenPassageId) {
      try {
        const chosenPassage = await passagesCol.findOne({
          _id: new ObjectId(body.chosenPassageId)
        });
        if (chosenPassage) {
          chosenFeatures = chosenPassage.features;
        }
      } catch {
        // Invalid ObjectId, skip
      }
    }
    
    if (body.candidatePassageIds && body.candidatePassageIds.length > 0) {
      try {
        const candidateIds = body.candidatePassageIds
          .map(id => {
            try { return new ObjectId(id); } catch { return null; }
          })
          .filter(Boolean) as ObjectId[];
        
        const candidates = await passagesCol.find({
          _id: { $in: candidateIds }
        }).toArray();
        
        candidateFeatures = candidates.map(p => ({
          passageId: p._id?.toString(),
          features: p.features
        }));
      } catch {
        // Error fetching candidates, skip
      }
    }
    
    // Create training event
    const event: TrainingEvent = {
      sessionId: body.sessionId,
      task: body.task,
      outcome: body.outcome,
      chosenPassageId: body.chosenPassageId,
      candidatePassageIds: body.candidatePassageIds || [],
      chosenFeatures,
      candidateFeatures,
      userFeedback: body.userFeedback,
      actionsTaken: body.actionsTaken,
      timeSpentMs: body.timeSpentMs,
      createdAt: new Date(),
    };
    
    const result = await trainingCol.insertOne(event);
    
    console.log(`📊 Logged feedback: session=${body.sessionId}, outcome=${body.outcome}`);
    
    return res.json({
      success: true,
      eventId: result.insertedId.toString(),
      message: 'Feedback logged successfully'
    });
    
  } catch (error: any) {
    console.error('❌ Feedback error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

// GET endpoint to retrieve training events (for XGBoost training)
router.get('/export', async (req: Request, res: Response) => {
  try {
    const trainingCol = getTrainingEventsCollection();
    
    const limit = Math.min(parseInt(req.query.limit as string) || 1000, 10000);
    const outcome = req.query.outcome as string;
    
    const filter: any = {};
    if (outcome) {
      filter.outcome = outcome;
    }
    
    const events = await trainingCol.find(filter)
      .sort({ createdAt: -1 })
      .limit(limit)
      .toArray();
    
    return res.json({
      count: events.length,
      events
    });
    
  } catch (error: any) {
    console.error('❌ Export error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

export default router;
