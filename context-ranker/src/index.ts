/**
 * Context Ranker - Main Entry Point
 * 
 * A service that stores and retrieves relevant context passages
 * for the backboard.io agent.
 */

import express from 'express';
import cors from 'cors';
import { config, validateConfig } from './utils/config';
import { connectDatabase, closeDatabase } from './services/database';
import contextRouter from './api/context';
import feedbackRouter from './api/feedback';

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// API Routes
app.use('/context', contextRouter);
app.use('/feedback', feedbackRouter);

// Error handler
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('❌ Error:', err.message);
  res.status(500).json({ error: err.message });
});

async function main() {
  // Validate configuration
  validateConfig();
  
  // Connect to database
  await connectDatabase();
  
  // Start server
  app.listen(config.port, () => {
    console.log(`🚀 Context Ranker running on http://localhost:${config.port}`);
    console.log(`   Environment: ${config.nodeEnv}`);
    console.log(`   XGBoost enabled: ${config.useXgboost}`);
  });
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n👋 Shutting down...');
  await closeDatabase();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await closeDatabase();
  process.exit(0);
});

main().catch((err) => {
  console.error('❌ Failed to start:', err);
  process.exit(1);
});
