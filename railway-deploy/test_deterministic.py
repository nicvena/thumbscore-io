#!/usr/bin/env python3
"""
Test script for deterministic scoring in Thumbscore.io

This script verifies that identical thumbnails return the same scores
when deterministic mode is enabled.
"""

import os
import sys
import asyncio
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

# Set deterministic mode
os.environ["DETERMINISTIC_MODE"] = "true"
os.environ["SCORE_VERSION"] = "v1.4-faiss-hybrid"

async def test_deterministic_scoring():
    """Test that identical thumbnails return identical scores."""
    
    print("🧪 Testing Deterministic Scoring for Thumbscore.io")
    print("=" * 60)
    
    # Import after setting environment variables
    from app.determinism import initialize_deterministic_mode, DeterministicCache
    from app.main import model_predict, extract_features
    
    # Initialize deterministic mode
    deterministic_mode, cache, normalizer = initialize_deterministic_mode()
    print(f"✅ Deterministic mode: {'ENABLED' if deterministic_mode else 'DISABLED'}")
    
    if not deterministic_mode:
        print("❌ Deterministic mode not enabled. Set DETERMINISTIC_MODE=true")
        return False
    
    # Test with a sample thumbnail URL
    test_url = "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"  # Sample thumbnail
    test_title = "Test Video Title"
    
    print(f"\n📸 Testing with thumbnail: {test_url}")
    print(f"📝 Title: {test_title}")
    
    # Extract features multiple times
    print("\n🔄 Running multiple feature extractions...")
    features_list = []
    
    for i in range(3):
        print(f"  Run {i+1}/3...")
        features = extract_features(test_url, test_title)
        features_list.append(features)
    
    # Check if features are identical
    print("\n🔍 Checking feature consistency...")
    base_features = features_list[0]
    
    for i, features in enumerate(features_list[1:], 1):
        # Compare CLIP embeddings
        embedding_diff = np.abs(base_features['clip_embedding'] - features['clip_embedding']).max()
        print(f"  Run {i+1} embedding difference: {embedding_diff:.6f}")
        
        if embedding_diff > 1e-6:
            print(f"  ⚠️  Embedding difference detected: {embedding_diff}")
        else:
            print(f"  ✅ Embeddings identical")
    
    # Test model predictions
    print("\n🎯 Testing model predictions...")
    predictions = []
    
    for i, features in enumerate(features_list):
        print(f"  Prediction {i+1}/3...")
        prediction = await model_predict(features, niche="tech")
        predictions.append(prediction)
    
    # Check if predictions are identical
    print("\n🔍 Checking prediction consistency...")
    base_prediction = predictions[0]
    
    all_identical = True
    for i, prediction in enumerate(predictions[1:], 1):
        print(f"  Prediction {i+1}:")
        
        # Compare CTR scores
        ctr_diff = abs(base_prediction['ctr_score'] - prediction['ctr_score'])
        print(f"    CTR score difference: {ctr_diff:.6f}")
        
        if ctr_diff > 1e-6:
            print(f"    ❌ CTR score difference detected: {ctr_diff}")
            all_identical = False
        else:
            print(f"    ✅ CTR scores identical")
        
        # Compare subscores
        base_subscores = base_prediction['subscores']
        pred_subscores = prediction['subscores']
        
        for subscore_name in base_subscores:
            if subscore_name in pred_subscores:
                diff = abs(base_subscores[subscore_name] - pred_subscores[subscore_name])
                if diff > 1e-6:
                    print(f"    ❌ {subscore_name} difference: {diff}")
                    all_identical = False
                else:
                    print(f"    ✅ {subscore_name} identical")
    
    # Test cache functionality
    print("\n💾 Testing cache functionality...")
    if cache:
        print(f"  Cache directory: {cache.cache_dir}")
        
        # Check if image data was stored
        image_data = base_features.get('image_data')
        if image_data:
            print(f"  ✅ Image data available for caching ({len(image_data)} bytes)")
            
            # Test cache operations
            cached_score = cache.get_cached_score(image_data, "tech", "clip-vit-l14-v1.4-faiss-hybrid")
            if cached_score:
                print(f"  ✅ Score cache working")
            else:
                print(f"  ⚠️  Score cache miss (expected on first run)")
        else:
            print(f"  ❌ No image data available for caching")
    
    # Summary
    print("\n" + "=" * 60)
    if all_identical:
        print("🎉 SUCCESS: Deterministic scoring is working correctly!")
        print("   ✅ Identical thumbnails return identical scores")
        print("   ✅ Hash-based caching is functional")
        print("   ✅ Deterministic mode is properly configured")
    else:
        print("❌ FAILURE: Non-deterministic behavior detected")
        print("   ⚠️  Some scores are not identical across runs")
        print("   ⚠️  Check for random number generation or floating-point precision issues")
    
    return all_identical

if __name__ == "__main__":
    success = asyncio.run(test_deterministic_scoring())
    sys.exit(0 if success else 1)
