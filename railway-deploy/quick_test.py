#!/usr/bin/env python3
"""
Quick test to verify scoring system is working
"""

import requests
import json

def test_scoring():
    print("🧪 Testing Thumbscore.io Scoring System")
    print("=" * 50)
    
    # Test data
    test_cases = [
        {
            "title": "Business Strategy Guide - INSANE Results!",
            "url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
            "expected_range": (70, 90)
        },
        {
            "title": "My Daily Vlog",
            "url": "https://i.ytimg.com/vi/jNQXAC9IVRw/maxresdefault.jpg", 
            "expected_range": (60, 80)
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📊 Test {i}: {test['title'][:30]}...")
        
        try:
            payload = {
                "title": test["title"],
                "thumbnails": [
                    {"id": f"test{i}", "url": test["url"]}
                ]
            }
            
            response = requests.post(
                "http://localhost:8000/v1/score", 
                json=payload, 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                score = data["thumbnails"][0]["ctr_score"]
                subscores = data["thumbnails"][0]["subscores"]
                
                print(f"   ✅ Score: {score}/100")
                print(f"   📈 Subscore Breakdown:")
                for key, value in subscores.items():
                    if isinstance(value, (int, float)):
                        print(f"      - {key}: {value}")
                
                # Check if score is realistic
                min_expected, max_expected = test["expected_range"]
                if min_expected <= score <= max_expected:
                    print(f"   ✅ Score in expected range ({min_expected}-{max_expected})")
                else:
                    print(f"   ⚠️ Score outside expected range ({min_expected}-{max_expected})")
                    
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout - server not responding")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test Complete")

if __name__ == "__main__":
    test_scoring()
