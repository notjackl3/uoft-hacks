/**
 * XGBoost Inference Integration
 * 
 * Calls Python inference script to get XGBoost scores for passages.
 * Enabled via USE_XGBOOST=true environment variable.
 */

import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { PassageFeatures } from '../models/schemas';
import { config } from '../utils/config';

const INFERENCE_SCRIPT = path.join(__dirname, '..', '..', 'scripts', 'infer_xgboost.py');
const VENV_PYTHON = path.join(__dirname, '..', '..', 'venv', 'bin', 'python3');

// Use venv Python if available, otherwise system Python
function getPythonPath(): string {
  if (fs.existsSync(VENV_PYTHON)) {
    return VENV_PYTHON;
  }
  return 'python3';
}

export interface XGBoostScores {
  scores: number[];
  error?: string;
}

/**
 * Get XGBoost scores for a list of passage features.
 * Falls back to empty scores on error.
 */
export async function getXGBoostScores(features: PassageFeatures[]): Promise<number[]> {
  if (!config.useXgboost) {
    return [];
  }

  return new Promise((resolve) => {
    const pythonPath = getPythonPath();
    const python = spawn(pythonPath, [INFERENCE_SCRIPT]);
    
    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        console.error(`❌ XGBoost inference failed (exit ${code}):`, stderr);
        resolve([]);
        return;
      }

      try {
        const result: XGBoostScores = JSON.parse(stdout);
        if (result.error) {
          console.error(`❌ XGBoost inference error:`, result.error);
          resolve([]);
          return;
        }
        resolve(result.scores || []);
      } catch (e) {
        console.error(`❌ Failed to parse XGBoost output:`, stdout);
        resolve([]);
      }
    });

    python.on('error', (err) => {
      console.error(`❌ Failed to spawn Python:`, err);
      resolve([]);
    });

    // Send features to stdin
    const input = JSON.stringify(features);
    python.stdin.write(input);
    python.stdin.end();

    // Timeout after 5 seconds
    setTimeout(() => {
      python.kill();
      console.error('❌ XGBoost inference timeout');
      resolve([]);
    }, 5000);
  });
}

/**
 * Check if XGBoost model is available.
 */
export function isXGBoostAvailable(): boolean {
  const modelPath = path.join(__dirname, '..', '..', 'models', 'ranker.json');
  try {
    require('fs').accessSync(modelPath);
    return true;
  } catch {
    return false;
  }
}
