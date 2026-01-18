/**
 * MongoDB Database Connection and Collection Access
 */

import { MongoClient, Db, Collection } from 'mongodb';
import { config } from '../utils/config';
import { Page, Passage, TrainingEvent } from '../models/schemas';

let client: MongoClient | null = null;
let db: Db | null = null;

export async function connectDatabase(): Promise<Db> {
  if (db) return db;
  
  console.log('🔌 Connecting to MongoDB...');
  client = new MongoClient(config.mongodbUri);
  await client.connect();
  db = client.db();
  
  // Create indexes for efficient querying
  await ensureIndexes();
  
  console.log('✅ Connected to MongoDB');
  return db;
}

async function ensureIndexes(): Promise<void> {
  if (!db) return;
  
  // Pages collection indexes
  const pages = db.collection<Page>('pages');
  await pages.createIndex({ url: 1 }, { unique: true });
  await pages.createIndex({ domain: 1 });
  await pages.createIndex({ contentHash: 1 });
  
  // Passages collection indexes
  const passages = db.collection<Passage>('passages');
  await passages.createIndex({ pageId: 1 });
  await passages.createIndex({ domain: 1 });
  await passages.createIndex({ url: 1 });
  // Text index for keyword search
  await passages.createIndex(
    { content: 'text', title: 'text' },
    { weights: { title: 2, content: 1 } }
  );
  
  // Training events collection indexes
  const events = db.collection<TrainingEvent>('training_events');
  await events.createIndex({ sessionId: 1 });
  await events.createIndex({ createdAt: -1 });
  
  console.log('📇 Database indexes ensured');
}

export function getCollection<T extends object>(name: string): Collection<T> {
  if (!db) {
    throw new Error('Database not connected. Call connectDatabase() first.');
  }
  return db.collection<T>(name);
}

export function getPagesCollection(): Collection<Page> {
  return getCollection<Page>('pages');
}

export function getPassagesCollection(): Collection<Passage> {
  return getCollection<Passage>('passages');
}

export function getTrainingEventsCollection(): Collection<TrainingEvent> {
  return getCollection<TrainingEvent>('training_events');
}

export async function closeDatabase(): Promise<void> {
  if (client) {
    await client.close();
    client = null;
    db = null;
    console.log('🔌 MongoDB connection closed');
  }
}
