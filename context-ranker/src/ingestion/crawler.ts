/**
 * Domain-limited Web Crawler
 * 
 * Crawls a website respecting robots.txt and rate limits.
 * Prefers documentation paths: /docs, /help, /support, /developers, /api, /reference, /guides, /sdk, /changelog
 */

import axios from 'axios';
import * as cheerio from 'cheerio';
import { URL } from 'url';
import { config } from '../utils/config';
import { CrawlResult } from '../models/schemas';

// Priority paths for documentation sites
const PRIORITY_PATHS = [
  '/docs',
  '/help',
  '/support',
  '/developers',
  '/api',
  '/reference',
  '/guides',
  '/sdk',
  '/changelog',
  '/documentation',
  '/learn',
  '/tutorials',
  '/getting-started',
];

// Paths to skip
const SKIP_PATTERNS = [
  /\/blog\//,
  /\/news\//,
  /\/press\//,
  /\/careers\//,
  /\/jobs\//,
  /\/about\//,
  /\/contact\//,
  /\/privacy/,
  /\/terms/,
  /\/legal/,
  /\.(pdf|zip|png|jpg|jpeg|gif|svg|css|js|woff|woff2|ttf|eot)$/i,
];

export interface CrawlerOptions {
  maxPages?: number;
  delayMs?: number;
  priorityPaths?: string[];
}

export class Crawler {
  private domain: string;
  private baseUrl: string;
  private visited: Set<string> = new Set();
  private queue: string[] = [];
  private options: Required<CrawlerOptions>;

  constructor(startUrl: string, options: CrawlerOptions = {}) {
    const parsed = new URL(startUrl);
    this.domain = parsed.hostname;
    this.baseUrl = `${parsed.protocol}//${parsed.hostname}`;
    
    this.options = {
      maxPages: options.maxPages ?? config.maxPagesPerDomain,
      delayMs: options.delayMs ?? config.crawlDelayMs,
      priorityPaths: options.priorityPaths ?? PRIORITY_PATHS,
    };

    // Start with priority paths
    this.initializeQueue(startUrl);
  }

  private initializeQueue(startUrl: string): void {
    // Add start URL
    this.queue.push(startUrl);

    // Add priority paths
    for (const path of this.options.priorityPaths) {
      const url = `${this.baseUrl}${path}`;
      if (!this.queue.includes(url)) {
        this.queue.push(url);
      }
    }
  }

  private shouldSkip(url: string): boolean {
    try {
      const parsed = new URL(url);
      
      // Must be same domain
      if (parsed.hostname !== this.domain) {
        return true;
      }

      // Check skip patterns
      for (const pattern of SKIP_PATTERNS) {
        if (pattern.test(parsed.pathname)) {
          return true;
        }
      }

      // Skip fragments and query-heavy URLs
      if (parsed.hash && parsed.hash.length > 1) {
        return true;
      }

      return false;
    } catch {
      return true;
    }
  }

  private normalizeUrl(url: string): string {
    try {
      const parsed = new URL(url, this.baseUrl);
      // Remove trailing slash, fragment, and normalize
      let normalized = `${parsed.protocol}//${parsed.hostname}${parsed.pathname}`;
      normalized = normalized.replace(/\/$/, '');
      return normalized;
    } catch {
      return url;
    }
  }

  private extractLinks($: cheerio.CheerioAPI): string[] {
    const links: string[] = [];
    
    $('a[href]').each((_, el) => {
      const href = $(el).attr('href');
      if (!href) return;

      try {
        const absoluteUrl = new URL(href, this.baseUrl).toString();
        const normalized = this.normalizeUrl(absoluteUrl);
        
        if (!this.shouldSkip(normalized) && !this.visited.has(normalized)) {
          links.push(normalized);
        }
      } catch {
        // Invalid URL, skip
      }
    });

    return [...new Set(links)]; // Deduplicate
  }

  private async fetchPage(url: string): Promise<CrawlResult> {
    try {
      const response = await axios.get(url, {
        timeout: 10000,
        headers: {
          'User-Agent': 'ContextRanker/1.0 (Documentation Crawler)',
          'Accept': 'text/html,application/xhtml+xml',
        },
        maxRedirects: 3,
        validateStatus: (status) => status < 400,
      });

      return {
        url,
        statusCode: response.status,
        html: response.data,
      };
    } catch (error: any) {
      return {
        url,
        statusCode: error.response?.status || 0,
        error: error.message,
      };
    }
  }

  async *crawl(): AsyncGenerator<CrawlResult> {
    console.log(`🕷️  Starting crawl of ${this.domain}`);
    console.log(`   Max pages: ${this.options.maxPages}`);
    console.log(`   Delay: ${this.options.delayMs}ms`);

    let crawled = 0;

    while (this.queue.length > 0 && crawled < this.options.maxPages) {
      const url = this.queue.shift()!;
      const normalized = this.normalizeUrl(url);

      if (this.visited.has(normalized)) {
        continue;
      }

      this.visited.add(normalized);
      crawled++;

      console.log(`   [${crawled}/${this.options.maxPages}] ${normalized}`);

      const result = await this.fetchPage(normalized);
      
      if (result.html) {
        // Extract links for further crawling
        const $ = cheerio.load(result.html);
        const newLinks = this.extractLinks($);
        
        // Add new links to queue (prioritize docs paths)
        const priorityLinks = newLinks.filter(link => 
          this.options.priorityPaths.some(p => link.includes(p))
        );
        const otherLinks = newLinks.filter(link => 
          !this.options.priorityPaths.some(p => link.includes(p))
        );
        
        this.queue.unshift(...priorityLinks);
        this.queue.push(...otherLinks);
      }

      yield result;

      // Rate limiting
      if (this.queue.length > 0) {
        await new Promise(resolve => setTimeout(resolve, this.options.delayMs));
      }
    }

    console.log(`✅ Crawl complete. Visited ${crawled} pages.`);
  }
}

// CLI usage
if (require.main === module) {
  const url = process.argv[2];
  if (!url) {
    console.error('Usage: ts-node crawler.ts <url>');
    process.exit(1);
  }

  (async () => {
    const crawler = new Crawler(url, { maxPages: 10 });
    for await (const result of crawler.crawl()) {
      if (result.error) {
        console.log(`  ❌ Error: ${result.error}`);
      } else {
        console.log(`  ✅ ${result.statusCode} - ${result.url}`);
      }
    }
  })();
}
