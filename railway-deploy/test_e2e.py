#!/usr/bin/env python3
"""
End-to-end test for the full scoring pipeline
"""
import requests
import json
import base64
import io
from PIL import Image

# Create a test image
def create_test_image():
    """Create a simple 400x300 test image with text-like patterns"""
    img = Image.new('RGB', (400, 300), color='#FF5733')
    
    # Add a simple rectangle to simulate a subject
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([150, 100, 250, 200], fill='#FFFFFF')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    
    return f"data:image/png;base64,{img_base64}"

def test_identical_images():
    """Test that identical images get identical scores"""
    print("🧪 Testing identical images scoring...")
    
    # Create one test image
    test_image_url = create_test_image()
    
    # Send request with two identical images
    payload = {
        "title": "Amazing Tech Tutorial - How To Fix Your PC",
        "thumbnails": [
            {"id": "thumb1", "url": test_image_url},
            {"id": "thumb2", "url": test_image_url}  # Same image
        ],
        "category": "tech"
    }
    
    print(f"📤 Sending request to http://localhost:8000/v1/score")
    print(f"   - Title: {payload['title']}")
    print(f"   - Thumbnails: 2 (identical images)")
    print(f"   - Category: {payload['category']}")
    print(f"   - Image data URL length: {len(test_image_url)} chars")
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/score",
            json=payload,
            timeout=20
        )
        
        print(f"\n📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS! Got response:")
            print(json.dumps(result, indent=2))
            
            # Check if scores are identical
            if len(result['thumbnails']) == 2:
                score1 = result['thumbnails'][0]['ctr_score']
                score2 = result['thumbnails'][1]['ctr_score']
                
                print(f"\n🎯 IDENTICAL IMAGE TEST:")
                print(f"   Thumbnail 1 Score: {score1}")
                print(f"   Thumbnail 2 Score: {score2}")
                
                if abs(score1 - score2) < 0.01:
                    print(f"   ✅ PASS: Scores are identical!")
                    return True
                else:
                    print(f"   ❌ FAIL: Scores differ by {abs(score1 - score2):.2f}")
                    return False
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ ERROR: Request timed out after 20 seconds")
        print(f"   Server may be hanging during processing")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 THUMBSCORE.IO - END-TO-END TEST")
    print("=" * 70)
    print()
    
    # Test server health first
    print("1️⃣ Checking server health...")
    try:
        health_response = requests.get("http://localhost:8000/", timeout=5)
        if health_response.status_code == 200:
            print(f"   ✅ Server is running: {health_response.json()['service']}")
        else:
            print(f"   ❌ Server returned {health_response.status_code}")
            exit(1)
    except Exception as e:
        print(f"   ❌ Server not accessible: {e}")
        exit(1)
    
    print()
    print("2️⃣ Testing identical images scoring...")
    success = test_identical_images()
    
    print()
    print("=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ TESTS FAILED - See errors above")
    print("=" * 70)

