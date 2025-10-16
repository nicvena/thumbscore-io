#!/usr/bin/env python3
"""
Test script to verify thumbnail scoring variations
This will test different types of thumbnails and show score ranges
"""

import requests
import json
import time

# Test different thumbnail URLs to see score variations
test_cases = [
    {
        "title": "How to Build a Website in 2024",
        "category": "tech", 
        "thumbnails": [
            {"id": "simple_text", "url": "https://via.placeholder.com/640x360/FF0000/FFFFFF?text=Simple+Text"},
            {"id": "complex_design", "url": "https://via.placeholder.com/640x360/0000FF/FFFFFF?text=Complex+Design+Here"},
            {"id": "minimal", "url": "https://via.placeholder.com/640x360/00FF00/000000?text=Min"}
        ]
    },
    {
        "title": "Ultimate Business Strategy Guide",
        "category": "business",
        "thumbnails": [
            {"id": "professional", "url": "https://via.placeholder.com/640x360/333333/FFFFFF?text=PROFESSIONAL"},
            {"id": "bright_colors", "url": "https://via.placeholder.com/640x360/FF6600/FFFFFF?text=BRIGHT+BUSINESS"},
            {"id": "simple_biz", "url": "https://via.placeholder.com/640x360/CCCCCC/000000?text=Business"}
        ]
    }
]

def test_scoring():
    base_url = "http://localhost:8001"
    
    print("🧪 Testing Thumbnail Scoring System")
    print("=" * 50)
    
    # Test health first
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Service Health: {health.json()['status']}")
    except Exception as e:
        print(f"❌ Service not accessible: {e}")
        return
    
    all_scores = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['title']} ({test_case['category']})")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{base_url}/v1/score",
                json=test_case,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"Winner: {result['winner_id']}")
                print("\nScores:")
                
                for thumb in result['thumbnails']:
                    score = thumb['ctr_score']
                    all_scores.append(score)
                    subscores = thumb['subscores']
                    
                    print(f"  {thumb['id']}: {score}/100")
                    print(f"    Similarity: {subscores['similarity']}")
                    print(f"    Clarity: {subscores['clarity']}")
                    print(f"    Contrast: {subscores['contrast_pop']}")
                    print(f"    Emotion: {subscores['emotion']}")
                    print(f"    Hierarchy: {subscores['hierarchy']}")
                    print()
                    
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        # Small delay between requests
        time.sleep(2)
    
    # Analyze score distribution
    if all_scores:
        print("\n📊 SCORE ANALYSIS")
        print("=" * 50)
        print(f"Total scores tested: {len(all_scores)}")
        print(f"Score range: {min(all_scores)} - {max(all_scores)}")
        print(f"Average score: {sum(all_scores)/len(all_scores):.1f}")
        print(f"Unique scores: {len(set(all_scores))}")
        
        # Check if we're getting variety
        if len(set(all_scores)) > len(all_scores) * 0.7:
            print("✅ GOOD: Scores show variety!")
        else:
            print("⚠️  WARNING: Scores may be too similar")
            
        # Check if we're not clustering around 84
        scores_around_84 = [s for s in all_scores if 82 <= s <= 86]
        if len(scores_around_84) < len(all_scores) * 0.5:
            print("✅ GOOD: Scores are not clustered around 84!")
        else:
            print("❌ ISSUE: Too many scores around 84")
    
    print(f"\n🌐 Frontend available at: http://localhost:3002")
    print("📝 Upload your own thumbnails to test!")

if __name__ == "__main__":
    test_scoring()