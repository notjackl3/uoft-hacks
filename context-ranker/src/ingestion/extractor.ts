/**
 * Content Extractor
 * 
 * Extracts main content from HTML, removing navigation, footer, and other noise.
 * Returns structured content: title, headings, mainText, codeBlocks
 */

import * as cheerio from 'cheerio';
import { ExtractedContent } from '../models/schemas';

// Elements to remove (navigation, footer, ads, etc.)
const REMOVE_SELECTORS = [
  'nav',
  'header',
  'footer',
  'aside',
  '.nav',
  '.navbar',
  '.navigation',
  '.header',
  '.footer',
  '.sidebar',
  '.menu',
  '.breadcrumb',
  '.breadcrumbs',
  '.toc',
  '.table-of-contents',
  '.advertisement',
  '.ad',
  '.ads',
  '.social',
  '.share',
  '.comments',
  '.comment-section',
  '[role="navigation"]',
  '[role="banner"]',
  '[role="contentinfo"]',
  '[aria-label="navigation"]',
  '[aria-label="footer"]',
  'script',
  'style',
  'noscript',
  'iframe',
  'svg',
];

// Selectors to try for main content (in priority order)
const MAIN_CONTENT_SELECTORS = [
  'main',
  'article',
  '[role="main"]',
  '.main-content',
  '.content',
  '.post-content',
  '.article-content',
  '.documentation',
  '.docs-content',
  '#content',
  '#main',
  '.markdown-body',
  '.prose',
];

export function extractContent(html: string, url?: string): ExtractedContent {
  const $ = cheerio.load(html);

  // Remove unwanted elements
  for (const selector of REMOVE_SELECTORS) {
    $(selector).remove();
  }

  // Extract title
  const title = $('title').text().trim() ||
                $('h1').first().text().trim() ||
                $('meta[property="og:title"]').attr('content') ||
                '';

  // Extract headings
  const headings: string[] = [];
  $('h1, h2, h3, h4').each((_, el) => {
    const text = $(el).text().trim();
    if (text && text.length > 2 && text.length < 200) {
      headings.push(text);
    }
  });

  // Extract code blocks
  const codeBlocks: string[] = [];
  $('pre, code').each((_, el) => {
    const code = $(el).text().trim();
    if (code && code.length > 10 && code.length < 5000) {
      codeBlocks.push(code);
    }
  });

  // Find main content area
  let mainElement: cheerio.Cheerio<any> | null = null;
  for (const selector of MAIN_CONTENT_SELECTORS) {
    const found = $(selector);
    if (found.length > 0) {
      mainElement = found.first();
      break;
    }
  }

  // Extract main text
  let mainText = '';
  if (mainElement) {
    mainText = extractTextFromElement($, mainElement);
  } else {
    // Fallback to body
    mainText = extractTextFromElement($, $('body'));
  }

  // Clean up the main text
  mainText = cleanText(mainText);

  return {
    title: cleanText(title),
    headings: headings.map(h => cleanText(h)).filter(h => h.length > 0),
    mainText,
    codeBlocks,
  };
}

function extractTextFromElement($: cheerio.CheerioAPI, element: cheerio.Cheerio<any>): string {
  // Clone to avoid modifying original
  const clone = element.clone();
  
  // Remove code blocks from text extraction (we handle them separately)
  clone.find('pre, code').remove();
  
  // Get text with some structure preserved
  let text = '';
  
  clone.find('p, li, td, th, div, span, h1, h2, h3, h4, h5, h6').each((_, el) => {
    const elText = $(el).text().trim();
    if (elText) {
      text += elText + '\n';
    }
  });

  // If no structured elements, fall back to all text
  if (!text.trim()) {
    text = clone.text();
  }

  return text;
}

function cleanText(text: string): string {
  return text
    // Normalize whitespace
    .replace(/[\t ]+/g, ' ')
    // Normalize newlines
    .replace(/\n\s*\n/g, '\n\n')
    // Remove excessive newlines
    .replace(/\n{3,}/g, '\n\n')
    // Trim
    .trim();
}

// Feature detection helpers
export function hasSteps(text: string): boolean {
  // Check for numbered lists or step indicators
  const patterns = [
    /step\s*\d/i,
    /^\s*\d+\.\s+/m,
    /first,?\s/i,
    /next,?\s/i,
    /then,?\s/i,
    /finally,?\s/i,
  ];
  return patterns.some(p => p.test(text));
}

export function hasCode(text: string, codeBlocks: string[]): boolean {
  return codeBlocks.length > 0 || /```|`[^`]+`/.test(text);
}

export function hasUIWords(text: string): boolean {
  const uiWords = [
    'click', 'button', 'menu', 'dropdown', 'select', 'checkbox',
    'radio', 'input', 'field', 'form', 'submit', 'dialog', 'modal',
    'tab', 'panel', 'sidebar', 'toolbar', 'icon', 'link', 'navigate',
  ];
  const lowerText = text.toLowerCase();
  return uiWords.some(word => lowerText.includes(word));
}

export function calculateMarketingScore(text: string): number {
  const marketingWords = [
    'amazing', 'incredible', 'revolutionary', 'best', 'leading',
    'powerful', 'seamless', 'innovative', 'cutting-edge', 'world-class',
    'enterprise', 'scale', 'transform', 'empower', 'unlock',
    'supercharge', 'turbocharge', 'game-changing', 'next-generation',
  ];
  
  const lowerText = text.toLowerCase();
  const wordCount = text.split(/\s+/).length;
  let marketingCount = 0;
  
  for (const word of marketingWords) {
    const regex = new RegExp(`\\b${word}\\b`, 'gi');
    const matches = text.match(regex);
    if (matches) {
      marketingCount += matches.length;
    }
  }
  
  // Normalize to 0-1 range
  return Math.min(1, marketingCount / Math.max(wordCount, 1) * 20);
}

export function calculateAuthorityTier(url: string): number {
  const path = new URL(url).pathname.toLowerCase();
  
  // Tier 1: Primary documentation
  if (/\/(docs|api|reference|sdk)/.test(path)) {
    return 1;
  }
  
  // Tier 2: Guides and tutorials
  if (/\/(guides|tutorials|learn|getting-started|quickstart)/.test(path)) {
    return 2;
  }
  
  // Tier 3: Everything else
  return 3;
}
