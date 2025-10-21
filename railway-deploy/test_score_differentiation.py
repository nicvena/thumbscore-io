#!/usr/bin/env python3
"""
Comprehensive test suite to verify score differentiation between good and bad thumbnails.
Tests the entire scoring pipeline with known quality levels.
"""

import os
import sys
import numpy as np
import logging
from typing import Dict, List, Any

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def create_test_thumbnail(quality_level: str) -> Dict[str, Any]:
    """
    Create synthetic test features for different quality levels.
    
    Args:
        quality_level: 'excellent', 'good', 'average', or 'poor'
        
    Returns:
        Dictionary of features matching the expected format
    """
    # Base embedding (normalized random vector)
    clip_embedding = np.random.randn(768).astype(np.float32)
    clip_embedding = clip_embedding / np.linalg.norm(clip_embedding)
    
    if quality_level == 'excellent':
        features = {
            'clip_embedding': clip_embedding,
            'ocr': {
                'word_count': 3,  # Minimal text
                'confidence': 0.95
            },
            'faces': {
                'dominant_face_size': 45,  # Large face (45% of image)
                'emotions': {
                    'happy': 0.85,  # Strong positive emotion
                    'surprise': 0.40,
                    'neutral': 0.05
                }
            },
            'colors': {
                'contrast': 150,  # High contrast
                'brightness': 130
            }
        }
    elif quality_level == 'good':
        features = {
            'clip_embedding': clip_embedding,
            'ocr': {
                'word_count': 5,  # Moderate text
                'confidence': 0.85
            },
            'faces': {
                'dominant_face_size': 30,  # Medium face (30% of image)
                'emotions': {
                    'happy': 0.60,  # Mild emotion
                    'surprise': 0.20,
                    'neutral': 0.20
                }
            },
            'colors': {
                'contrast': 110,  # Good contrast
                'brightness': 115
            }
        }
    elif quality_level == 'average':
        features = {
            'clip_embedding': clip_embedding,
            'ocr': {
                'word_count': 8,  # Moderate-high text
                'confidence': 0.75
            },
            'faces': {
                'dominant_face_size': 18,  # Small face (18% of image)
                'emotions': {
                    'happy': 0.30,  # Weak emotion
                    'surprise': 0.10,
                    'neutral': 0.60
                }
            },
            'colors': {
                'contrast': 85,  # Okay contrast
                'brightness': 100
            }
        }
    else:  # poor
        features = {
            'clip_embedding': clip_embedding,
            'ocr': {
                'word_count': 15,  # Too much text
                'confidence': 0.60
            },
            'faces': {
                'dominant_face_size': 5,  # Very small/no face (5% of image)
                'emotions': {
                    'happy': 0.10,  # No emotion
                    'surprise': 0.05,
                    'neutral': 0.85
                }
            },
            'colors': {
                'contrast': 50,  # Low contrast
                'brightness': 80
            }
        }
    
    return features

def test_quality_differentiation():
    """Test that the scoring system differentiates between quality levels"""
    
    print("=" * 80)
    print("🎯 SCORE DIFFERENTIATION TEST SUITE")
    print("=" * 80)
    print()
    
    try:
        from app.main import model_predict
        
        quality_levels = ['excellent', 'good', 'average', 'poor']
        niches = ['gaming', 'tech', 'entertainment']
        
        # Expected score ranges
        expected_ranges = {
            'excellent': (85, 95),
            'good': (75, 85),
            'average': (60, 75),
            'poor': (40, 60)
        }
        
        all_results = []
        
        print("📊 TESTING SCORE DIFFERENTIATION ACROSS QUALITY LEVELS")
        print("-" * 80)
        
        # Test each quality level in each niche
        for niche in niches:
            print(f"\n🎮 Testing Niche: {niche.upper()}")
            print("-" * 80)
            
            niche_results = []
            
            for quality in quality_levels:
                # Create test features
                features = create_test_thumbnail(quality)
                
                # Get prediction
                prediction = model_predict(features, niche)
                
                ctr_score = prediction['ctr_score']
                subscores = prediction['subscores']
                raw_score = prediction.get('raw_ctr_score', 0)
                similarity_source = prediction.get('similarity_source', 'unknown')
                
                # Store results
                result = {
                    'niche': niche,
                    'quality': quality,
                    'ctr_score': ctr_score,
                    'raw_score': raw_score,
                    'similarity': subscores.get('similarity', 0),
                    'clarity': subscores.get('clarity', 0),
                    'contrast_pop': subscores.get('contrast_pop', 0),
                    'emotion': subscores.get('emotion', 0),
                    'similarity_source': similarity_source,
                    'expected_range': expected_ranges[quality]
                }
                
                niche_results.append(result)
                all_results.append(result)
                
                # Check if score is in expected range
                expected_min, expected_max = expected_ranges[quality]
                in_range = expected_min <= ctr_score <= expected_max
                status = "✅" if in_range else "⚠️"
                
                print(f"{status} {quality.upper():10s}: {ctr_score:5.1f}% (expected {expected_min}-{expected_max}%) | raw={raw_score:.1f}")
                print(f"   Subscores: sim={subscores.get('similarity'):3d} clarity={subscores.get('clarity'):3d} color={subscores.get('contrast_pop'):3d} emotion={subscores.get('emotion'):3d}")
                print(f"   Similarity source: {similarity_source}")
            
            # Calculate spread for this niche
            scores = [r['ctr_score'] for r in niche_results]
            spread = max(scores) - min(scores)
            
            print(f"\n   Score Range: {min(scores):.1f}% - {max(scores):.1f}%")
            print(f"   Spread: {spread:.1f}%")
            
            if spread >= 30:
                print(f"   ✅ Excellent differentiation (≥30% spread)")
            elif spread >= 20:
                print(f"   ✅ Good differentiation (≥20% spread)")
            else:
                print(f"   ⚠️  Limited differentiation (<20% spread)")
        
        # Overall analysis
        print("\n" + "=" * 80)
        print("📈 OVERALL DIFFERENTIATION ANALYSIS")
        print("=" * 80)
        
        # Group by quality level
        by_quality = {}
        for result in all_results:
            quality = result['quality']
            if quality not in by_quality:
                by_quality[quality] = []
            by_quality[quality].append(result['ctr_score'])
        
        print("\n🎯 Average Scores by Quality Level:")
        print("-" * 80)
        
        quality_avgs = {}
        for quality in quality_levels:
            scores = by_quality.get(quality, [])
            avg_score = np.mean(scores) if scores else 0
            std_dev = np.std(scores) if scores else 0
            expected_min, expected_max = expected_ranges[quality]
            
            quality_avgs[quality] = avg_score
            
            in_range = expected_min <= avg_score <= expected_max
            status = "✅" if in_range else "⚠️"
            
            print(f"{status} {quality.upper():10s}: {avg_score:5.1f}% ± {std_dev:4.1f}% (expected {expected_min}-{expected_max}%)")
        
        # Calculate overall spread
        overall_spread = quality_avgs['excellent'] - quality_avgs['poor']
        
        print(f"\n📊 Overall Metrics:")
        print(f"   Excellent → Poor Spread: {overall_spread:.1f}%")
        print(f"   Excellent avg: {quality_avgs['excellent']:.1f}%")
        print(f"   Good avg: {quality_avgs['good']:.1f}%")
        print(f"   Average avg: {quality_avgs['average']:.1f}%")
        print(f"   Poor avg: {quality_avgs['poor']:.1f}%")
        
        # Verify ordering
        print(f"\n🔍 Quality Ordering Verification:")
        ordering_correct = (
            quality_avgs['excellent'] > quality_avgs['good'] >
            quality_avgs['average'] > quality_avgs['poor']
        )
        
        if ordering_correct:
            print(f"   ✅ Correct ordering: Excellent > Good > Average > Poor")
        else:
            print(f"   ❌ Incorrect ordering detected!")
        
        # Check spread
        print(f"\n🎯 Differentiation Assessment:")
        if overall_spread >= 30:
            print(f"   ✅ EXCELLENT differentiation ({overall_spread:.1f}% spread)")
        elif overall_spread >= 20:
            print(f"   ✅ GOOD differentiation ({overall_spread:.1f}% spread)")
        elif overall_spread >= 10:
            print(f"   ⚠️  MODERATE differentiation ({overall_spread:.1f}% spread)")
        else:
            print(f"   ❌ POOR differentiation ({overall_spread:.1f}% spread)")
        
        # Count how many scores are in expected ranges
        in_range_count = sum(1 for r in all_results if r['expected_range'][0] <= r['ctr_score'] <= r['expected_range'][1])
        total_count = len(all_results)
        
        print(f"\n✅ Test Results:")
        print(f"   {in_range_count}/{total_count} scores in expected ranges ({100*in_range_count/total_count:.1f}%)")
        
        # Final verdict
        print("\n" + "=" * 80)
        if ordering_correct and overall_spread >= 20 and in_range_count/total_count >= 0.7:
            print("🎉 ✅ SCORE DIFFERENTIATION TEST: PASSED")
            print("   The scoring system successfully differentiates quality levels!")
            return True
        else:
            print("⚠️  ❌ SCORE DIFFERENTIATION TEST: NEEDS IMPROVEMENT")
            print("   The scoring system may need tuning for better differentiation.")
            return False
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detailed_breakdown():
    """Test with detailed breakdown of scoring components"""
    
    print("\n" + "=" * 80)
    print("🔬 DETAILED SCORING BREAKDOWN TEST")
    print("=" * 80)
    
    try:
        from app.main import model_predict
        
        # Test one excellent and one poor thumbnail with full details
        print("\n📊 EXCELLENT vs POOR - Detailed Comparison")
        print("-" * 80)
        
        for quality in ['excellent', 'poor']:
            print(f"\n{'🌟' if quality == 'excellent' else '💀'} {quality.upper()} Quality Thumbnail:")
            print("-" * 40)
            
            features = create_test_thumbnail(quality)
            prediction = model_predict(features, 'tech')
            
            ctr_score = prediction['ctr_score']
            raw_score = prediction.get('raw_ctr_score', 0)
            subscores = prediction['subscores']
            weights = prediction.get('weights_used', {})
            
            print(f"Final CTR Score: {ctr_score:.1f}%")
            print(f"Raw Score (pre-amplification): {raw_score:.1f}%")
            print(f"Amplification: {ctr_score - raw_score:+.1f}%")
            
            print(f"\nSubscores:")
            for key, value in subscores.items():
                weight = weights.get(key, 0) if key in weights else 0
                contribution = value * weight if weight > 0 else 0
                print(f"  {key:20s}: {value:3d} (weight: {weight:.2f}, contribution: {contribution:5.1f})")
            
            print(f"\nInput Features:")
            print(f"  Word count: {features['ocr']['word_count']}")
            print(f"  Face size: {features['faces']['dominant_face_size']}%")
            print(f"  Contrast: {features['colors']['contrast']}")
            print(f"  Happy emotion: {features['faces']['emotions']['happy']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Detailed breakdown test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consistency():
    """Test that scores are consistent for the same quality level"""
    
    print("\n" + "=" * 80)
    print("🔄 SCORE CONSISTENCY TEST")
    print("=" * 80)
    
    try:
        from app.main import model_predict
        
        quality = 'good'
        niche = 'tech'
        num_runs = 5
        
        print(f"\nTesting consistency for {quality.upper()} quality in {niche} niche")
        print(f"Running {num_runs} iterations with different random embeddings...")
        print("-" * 80)
        
        scores = []
        for i in range(num_runs):
            features = create_test_thumbnail(quality)
            prediction = model_predict(features, niche)
            score = prediction['ctr_score']
            scores.append(score)
            print(f"Run {i+1}: {score:.1f}%")
        
        avg_score = np.mean(scores)
        std_dev = np.std(scores)
        score_range = max(scores) - min(scores)
        
        print(f"\n📊 Consistency Metrics:")
        print(f"   Average: {avg_score:.1f}%")
        print(f"   Std Dev: {std_dev:.1f}%")
        print(f"   Range: {score_range:.1f}%")
        
        if std_dev <= 5:
            print(f"   ✅ Excellent consistency (std dev ≤ 5%)")
            return True
        elif std_dev <= 10:
            print(f"   ✅ Good consistency (std dev ≤ 10%)")
            return True
        else:
            print(f"   ⚠️  High variance detected (std dev > 10%)")
            return False
        
    except Exception as e:
        print(f"\n❌ Consistency test failed: {e}")
        return False

if __name__ == "__main__":
    print("Score Differentiation Test Suite")
    print("Testing the scoring system's ability to differentiate quality levels")
    print()
    
    # Run all tests
    tests = [
        ("Quality Differentiation", test_quality_differentiation),
        ("Detailed Breakdown", test_detailed_breakdown),
        ("Score Consistency", test_consistency)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test failed with exception: {e}")
            results.append((name, False))
    
    # Final summary
    print("\n" + "=" * 80)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Score differentiation is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Review the results above.")
        sys.exit(1)

