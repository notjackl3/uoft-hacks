/**
 * Ingestion Pipeline
 * 
 * Orchestrates: crawl -> extract -> chunk -> embed -> store
 */

import { ObjectId } from 'mongodb';
import crypto from 'crypto';
import { Crawler, CrawlerOptions } from './crawler';
import { 
  extractContent, 
  hasSteps, 
  hasCode, 
  hasUIWords, 
  calculateMarketingScore,
  calculateAuthorityTier 
} from './extractor';
import { chunkText, countTokens } from './chunker';
import { embedTexts } from '../services/embeddings';
import { 
  connectDatabase, 
  getPagesCollection, 
  getPassagesCollection,
  closeDatabase 
} from '../services/database';
import { Page, Passage, PassageFeatures } from '../models/schemas';

export interface IngestOptions extends CrawlerOptions {
  batchSize?: number;  // Batch size for embedding API calls
}

export async function ingestSite(startUrl: string, options: IngestOptions = {}): Promise<{
  pagesProcessed: number;
  passagesCreated: number;
  errors: string[];
}> {
  const batchSize = options.batchSize ?? 10;
  const errors: string[] = [];
  let pagesProcessed = 0;
  let passagesCreated = 0;

  console.log(`\n📥 Starting ingestion of ${startUrl}`);
  console.log('='.repeat(50));

  await connectDatabase();
  const pagesCol = getPagesCollection();
  const passagesCol = getPassagesCollection();

  const crawler = new Crawler(startUrl, options);

  // Collect passages to embed in batches
  let passageBatch: {
    passage: Omit<Passage, 'embedding'>;
    text: string;
  }[] = [];

  async function flushBatch() {
    if (passageBatch.length === 0) return;

    try {
      console.log(`   🔢 Embedding batch of ${passageBatch.length} passages...`);
      
      const texts = passageBatch.map(p => p.text);
      const embeddings = await embedTexts(texts);

      const passagesToInsert: Passage[] = passageBatch.map((p, i) => ({
        ...p.passage,
        embedding: embeddings[i],
      } as Passage));

      await passagesCol.insertMany(passagesToInsert);
      passagesCreated += passagesToInsert.length;
      console.log(`   ✅ Stored ${passagesToInsert.length} passages`);
    } catch (err: any) {
      console.error(`   ❌ Batch embedding failed: ${err.message}`);
      errors.push(`Batch embedding: ${err.message}`);
    }

    passageBatch = [];
  }

  for await (const result of crawler.crawl()) {
    if (result.error || !result.html) {
      if (result.error) {
        errors.push(`${result.url}: ${result.error}`);
      }
      continue;
    }

    try {
      // Extract content
      const content = extractContent(result.html, result.url);
      
      if (!content.mainText || content.mainText.length < 100) {
        console.log(`   ⏭️  Skipping (no content): ${result.url}`);
        continue;
      }

      // Create content hash for deduplication
      const contentHash = crypto
        .createHash('md5')
        .update(content.mainText)
        .digest('hex');

      // Check if page already exists with same content
      const existingPage = await pagesCol.findOne({ 
        url: result.url,
        contentHash 
      });

      if (existingPage) {
        console.log(`   ⏭️  Skipping (unchanged): ${result.url}`);
        continue;
      }

      // Parse URL
      const parsedUrl = new URL(result.url);

      // Create/update page document
      const pageDoc: Page = {
        url: result.url,
        domain: parsedUrl.hostname,
        path: parsedUrl.pathname,
        title: content.title,
        headings: content.headings,
        crawledAt: new Date(),
        contentHash,
        statusCode: result.statusCode,
        passageCount: 0, // Will update after chunking
      };

      // Upsert page
      const pageResult = await pagesCol.updateOne(
        { url: result.url },
        { $set: pageDoc },
        { upsert: true }
      );
      
      const pageId = pageResult.upsertedId || 
        (await pagesCol.findOne({ url: result.url }))?._id;

      if (!pageId) {
        throw new Error('Failed to get page ID');
      }

      // Delete old passages for this page (if re-crawling)
      await passagesCol.deleteMany({ pageId });

      // Chunk the content
      const chunks = chunkText(content.mainText);
      console.log(`   📄 ${result.url} -> ${chunks.length} chunks`);

      // Calculate features and queue for embedding
      const authorityTier = calculateAuthorityTier(result.url);
      
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        
        const features: PassageFeatures = {
          hasSteps: hasSteps(chunk.text),
          hasCode: hasCode(chunk.text, content.codeBlocks),
          hasUIWords: hasUIWords(chunk.text),
          marketingScore: calculateMarketingScore(chunk.text),
          authorityTier,
          freshnessDays: 0, // Will be updated based on last-modified header
          chunkPosition: i,
          totalChunks: chunks.length,
          tokenCount: chunk.tokenCount,
        };

        const passage: Omit<Passage, 'embedding'> = {
          pageId: pageId as ObjectId,
          url: result.url,
          domain: parsedUrl.hostname,
          title: content.title,
          content: chunk.text,
          features,
          createdAt: new Date(),
          updatedAt: new Date(),
        };

        passageBatch.push({
          passage,
          text: `${content.title}\n\n${chunk.text}`,
        });

        // Flush batch if full
        if (passageBatch.length >= batchSize) {
          await flushBatch();
        }
      }

      // Update page with passage count
      await pagesCol.updateOne(
        { _id: pageId },
        { $set: { passageCount: chunks.length } }
      );

      pagesProcessed++;

    } catch (err: any) {
      console.error(`   ❌ Error processing ${result.url}: ${err.message}`);
      errors.push(`${result.url}: ${err.message}`);
    }
  }

  // Flush remaining batch
  await flushBatch();

  console.log('\n' + '='.repeat(50));
  console.log(`✅ Ingestion complete!`);
  console.log(`   Pages processed: ${pagesProcessed}`);
  console.log(`   Passages created: ${passagesCreated}`);
  if (errors.length > 0) {
    console.log(`   Errors: ${errors.length}`);
  }

  return { pagesProcessed, passagesCreated, errors };
}

// CLI usage
if (require.main === module) {
  const url = process.argv[2];
  const maxPages = parseInt(process.argv[3] || '20', 10);

  if (!url) {
    console.error('Usage: ts-node pipeline.ts <url> [maxPages]');
    console.error('Example: ts-node pipeline.ts https://stripe.com/docs 50');
    process.exit(1);
  }

  (async () => {
    try {
      const result = await ingestSite(url, { maxPages });
      
      if (result.errors.length > 0) {
        console.log('\n❌ Errors encountered:');
        result.errors.slice(0, 10).forEach(e => console.log(`   - ${e}`));
        if (result.errors.length > 10) {
          console.log(`   ... and ${result.errors.length - 10} more`);
        }
      }
    } catch (err) {
      console.error('Fatal error:', err);
      process.exit(1);
    } finally {
      await closeDatabase();
    }
  })();
}
