#!/usr/bin/env python3
"""
XGBoost Inference Script for Context Ranker

Takes feature vectors via stdin (JSON) and returns scores.
Used by the Node.js server when USE_XGBOOST=true.

Usage:
    echo '[{"hasSteps": true, "hasCode": false, ...}]' | python scripts/infer_xgboost.py
    
Returns:
    JSON array of scores
"""

import sys
import json
import os

try:
    import xgboost as xgb
    import numpy as np
except ImportError:
    print(json.dumps({"error": "XGBoost or numpy not installed"}))
    sys.exit(1)

# Feature names (must match training)
FEATURE_NAMES = [
    'hasSteps',
    'hasCode', 
    'hasUIWords',
    'marketingScore',
    'authorityTier',
    'freshnessDays',
    'chunkPosition',
    'totalChunks',
    'tokenCount',
]

def extract_features(features_dict: dict) -> list:
    """Extract feature vector from features dictionary."""
    return [
        1.0 if features_dict.get('hasSteps', False) else 0.0,
        1.0 if features_dict.get('hasCode', False) else 0.0,
        1.0 if features_dict.get('hasUIWords', False) else 0.0,
        float(features_dict.get('marketingScore', 0.5)),
        float(features_dict.get('authorityTier', 2)) / 3.0,
        min(float(features_dict.get('freshnessDays', 0)) / 365.0, 1.0),
        float(features_dict.get('chunkPosition', 0)) / max(float(features_dict.get('totalChunks', 1)), 1.0),
        min(float(features_dict.get('totalChunks', 1)) / 50.0, 1.0),
        min(float(features_dict.get('tokenCount', 300)) / 500.0, 1.0),
    ]

def load_model():
    """Load the trained XGBoost model."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ranker.json')
    
    if not os.path.exists(model_path):
        return None, f"Model not found at {model_path}"
    
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def predict(model, features_list: list) -> list:
    """Predict scores for a list of feature dictionaries."""
    X = np.array([extract_features(f) for f in features_list])
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    scores = model.predict(dmatrix)
    return scores.tolist()

def main():
    # Load model
    model, error = load_model()
    
    if error:
        print(json.dumps({"error": error}))
        sys.exit(1)
    
    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}))
        sys.exit(1)
    
    if not isinstance(input_data, list):
        print(json.dumps({"error": "Input must be a JSON array of feature objects"}))
        sys.exit(1)
    
    if len(input_data) == 0:
        print(json.dumps({"scores": []}))
        return
    
    # Predict
    try:
        scores = predict(model, input_data)
        print(json.dumps({"scores": scores}))
    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
        sys.exit(1)

if __name__ == '__main__':
    main()
