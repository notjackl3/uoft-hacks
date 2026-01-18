#!/usr/bin/env python3
"""
XGBoost Training Script for Context Ranker

Reads training_events from MongoDB and trains a ranking model.
Outputs model file to models/ranker.json

Usage:
    python scripts/train_xgboost.py
    
Environment:
    MONGODB_URI - MongoDB connection string
"""

import os
import json
import numpy as np
from pymongo import MongoClient
from datetime import datetime

try:
    import xgboost as xgb
except ImportError:
    print("❌ XGBoost not installed. Run: pip install xgboost")
    exit(1)

# Feature names for the model
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

def load_training_data(mongodb_uri: str):
    """Load training events from MongoDB."""
    print("📥 Loading training data from MongoDB...")
    
    client = MongoClient(mongodb_uri)
    db = client.get_database()
    
    events = list(db.training_events.find())
    print(f"   Found {len(events)} training events")
    
    client.close()
    return events

def extract_features(features_dict: dict) -> list:
    """Extract feature vector from features dictionary."""
    return [
        1.0 if features_dict.get('hasSteps', False) else 0.0,
        1.0 if features_dict.get('hasCode', False) else 0.0,
        1.0 if features_dict.get('hasUIWords', False) else 0.0,
        float(features_dict.get('marketingScore', 0.5)),
        float(features_dict.get('authorityTier', 2)) / 3.0,  # Normalize to 0-1
        min(float(features_dict.get('freshnessDays', 0)) / 365.0, 1.0),  # Cap at 1 year
        float(features_dict.get('chunkPosition', 0)) / max(float(features_dict.get('totalChunks', 1)), 1.0),
        min(float(features_dict.get('totalChunks', 1)) / 50.0, 1.0),  # Normalize
        min(float(features_dict.get('tokenCount', 300)) / 500.0, 1.0),  # Normalize
    ]

def prepare_training_data(events: list):
    """
    Prepare XGBoost training data from events.
    
    For ranking, we create pairs:
    - Positive: chosen passage (if outcome was success)
    - Negative: other candidate passages
    """
    X = []  # Feature vectors
    y = []  # Labels (1 = chosen/positive, 0 = not chosen)
    groups = []  # Query groups for ranking
    
    for event in events:
        if not event.get('candidateFeatures'):
            continue
            
        outcome = event.get('outcome', 'failure')
        chosen_id = event.get('chosenPassageId')
        
        # Only use success events with chosen passages for positive examples
        if outcome == 'success' and chosen_id and event.get('chosenFeatures'):
            group_size = 0
            
            # Add chosen passage as positive example
            features = extract_features(event['chosenFeatures'])
            X.append(features)
            y.append(1.0)
            group_size += 1
            
            # Add candidate passages as negative examples
            for candidate in event['candidateFeatures']:
                if candidate.get('passageId') != chosen_id:
                    features = extract_features(candidate.get('features', {}))
                    X.append(features)
                    y.append(0.0)
                    group_size += 1
            
            if group_size > 1:
                groups.append(group_size)
        
        # For failure events, all passages are negative
        elif outcome == 'failure' and event.get('candidateFeatures'):
            group_size = 0
            for candidate in event['candidateFeatures']:
                features = extract_features(candidate.get('features', {}))
                X.append(features)
                y.append(0.0)
                group_size += 1
            
            if group_size > 0:
                groups.append(group_size)
    
    return np.array(X), np.array(y), groups

def train_model(X: np.ndarray, y: np.ndarray, groups: list):
    """Train XGBoost ranking model."""
    print(f"🎯 Training XGBoost model...")
    print(f"   Samples: {len(X)}")
    print(f"   Query groups: {len(groups)}")
    print(f"   Features: {len(FEATURE_NAMES)}")
    
    if len(X) < 10:
        print("❌ Not enough training data (need at least 10 samples)")
        return None
    
    # Create DMatrix for ranking
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    dtrain.set_group(groups)
    
    # XGBoost parameters for ranking
    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg',
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }
    
    # Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        verbose_eval=10,
    )
    
    return model

def save_model(model, output_path: str):
    """Save model to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    print(f"✅ Model saved to {output_path}")

def main():
    # Get MongoDB URI from environment
    mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/context-ranker')
    
    # Load training data
    events = load_training_data(mongodb_uri)
    
    if not events:
        print("❌ No training events found. Log some feedback first!")
        print("   POST /feedback with outcome and chosen passages")
        return
    
    # Prepare data
    X, y, groups = prepare_training_data(events)
    
    if len(X) == 0:
        print("❌ No valid training samples extracted.")
        print("   Make sure events have candidateFeatures and chosenFeatures")
        return
    
    print(f"📊 Training data prepared:")
    print(f"   Total samples: {len(X)}")
    print(f"   Positive samples: {int(y.sum())}")
    print(f"   Negative samples: {int(len(y) - y.sum())}")
    
    # Train model
    model = train_model(X, y, groups)
    
    if model:
        # Save model
        output_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ranker.json')
        save_model(model, output_path)
        
        # Print feature importance
        print("\n📈 Feature Importance:")
        importance = model.get_score(importance_type='gain')
        for name, score in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"   {name}: {score:.4f}")

if __name__ == '__main__':
    main()
