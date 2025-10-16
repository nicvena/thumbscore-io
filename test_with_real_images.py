#!/usr/bin/env python3
"""
Test scoring with real local images
"""

import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import os

def create_test_image(text, bg_color, text_color, size=(640, 360)):
    """Create a simple test thumbnail image"""
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Get text size and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) / 2
    y = (size[1] - text_height) / 2
    
    draw.text((x, y), text, fill=text_color, font=font)
    return img

def image_to_base64_url(image):
    """Convert PIL image to base64 data URL"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def test_with_local_images():
    print("🧪 Testing with Generated Images")
    print("=" * 50)
    
    # Create different test thumbnails
    test_images = [
        {
            "id": "high_contrast",
            "image": create_test_image("CLICK HERE!", (255, 0, 0), (255, 255, 255)),
            "description": "High contrast red background"
        },
        {
            "id": "low_contrast", 
            "image": create_test_image("boring text", (200, 200, 200), (150, 150, 150)),
            "description": "Low contrast gray"
        },
        {
            "id": "minimal_text",
            "image": create_test_image("GO", (0, 255, 0), (0, 0, 0)),
            "description": "Minimal text, bright green"
        },
        {
            "id": "long_text",
            "image": create_test_image("Very Long Title That Takes Up Space", (0, 0, 255), (255, 255, 255)),
            "description": "Long text, blue background"
        }
    ]
    
    # Convert to base64 URLs
    thumbnails = []
    for img_data in test_images:
        url = image_to_base64_url(img_data["image"])
        thumbnails.append({
            "id": img_data["id"],
            "url": url
        })
        print(f"✅ Created {img_data['id']}: {img_data['description']}")
    
    # Test scoring
    test_case = {
        "title": "Amazing Tutorial You Must Watch",
        "category": "tech",
        "thumbnails": thumbnails
    }
    
    print(f"\n🎯 Testing {len(thumbnails)} thumbnails...")
    
    try:
        response = requests.post(
            "http://localhost:8001/v1/score",
            json=test_case,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n🏆 Winner: {result['winner_id']}")
            print(f"📊 Niche: {result['niche']}")
            
            scores = []
            print(f"\n📋 DETAILED SCORES:")
            print("-" * 60)
            
            for thumb in result['thumbnails']:
                score = thumb['ctr_score']
                scores.append(score)
                subscores = thumb['subscores']
                
                print(f"\n🖼️  {thumb['id'].upper()}: {score}/100")
                print(f"   Similarity: {subscores['similarity']}/100")
                print(f"   Clarity: {subscores['clarity']}/100") 
                print(f"   Contrast: {subscores['contrast_pop']}/100")
                print(f"   Emotion: {subscores['emotion']}/100")
                print(f"   Hierarchy: {subscores['hierarchy']}/100")
                
                # Show insights
                if thumb.get('insights'):
                    print(f"   💡 Insights: {thumb['insights'][:2]}")  # First 2 insights
            
            # Analyze results
            print(f"\n📈 ANALYSIS:")
            print(f"   Score range: {min(scores)} - {max(scores)}")
            print(f"   Average: {sum(scores)/len(scores):.1f}")
            print(f"   Unique scores: {len(set(scores))}/{len(scores)}")
            
            if len(set(scores)) == len(scores):
                print("   ✅ All scores are unique!")
            elif len(set(scores)) > len(scores) * 0.75:
                print("   ✅ Good score variety!")
            else:
                print("   ⚠️  Limited score variety")
                
            # Check if scores are realistic
            if max(scores) - min(scores) > 10:
                print("   ✅ Good score spread!")
            else:
                print("   ⚠️  Scores may be too similar")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text[:500])
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_with_local_images()