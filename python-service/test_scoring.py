#!/usr/bin/env python3
"""
Consistency Test Script for V1 Simplified Scoring System

Tests that the simplified scoring system provides:
1. Consistent scores (±2 points) for the same thumbnail
2. Different scores for different thumbnails
3. Proper score distribution across all niches

Usage:
    python test_scoring.py [--niche gaming] [--runs 5] [--test-image path/to/image.jpg]
"""

import os
import argparse
import asyncio
import requests
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scoring_simple import test_consistency, score_thumbnail, compare_thumbnails

def load_test_image(image_path: str) -> bytes:
    """Load test image from file or URL"""
    if image_path.startswith('http'):
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        return response.content
    else:
        with open(image_path, 'rb') as f:
            return f.read()

def create_sample_test_image() -> bytes:
    """Create a simple test image if no image provided"""
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    # Create a 1280x720 thumbnail-like image
    img = Image.new('RGB', (1280, 720), color='#2E86AB')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        font = ImageFont.truetype("Arial.ttf", 80)
    except:
        font = ImageFont.load_default()
    
    # White text with black outline
    text = "TEST THUMBNAIL"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (1280 - text_width) // 2
    y = (720 - text_height) // 2
    
    # Draw text with outline
    for adj in range(3):
        draw.text((x-adj, y-adj), text, font=font, fill='black')
        draw.text((x+adj, y-adj), text, font=font, fill='black')
        draw.text((x-adj, y+adj), text, font=font, fill='black')
        draw.text((x+adj, y+adj), text, font=font, fill='black')
    draw.text((x, y), text, font=font, fill='white')
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG', quality=85)
    return img_buffer.getvalue()

async def test_single_niche_consistency(image_data: bytes, niche: str, runs: int = 5) -> Dict[str, Any]:
    """Test scoring consistency for a single niche"""
    print(f"\n🔍 Testing {niche} niche consistency ({runs} runs)...")
    
    result = test_consistency(image_data, niche, runs)
    
    status = "✅ PASS" if result["test_passed"] else "❌ FAIL"
    print(f"   {status} - {result['recommendation']}")
    print(f"   Mean: {result['mean_score']}, StdDev: {result['std_deviation']}, Range: {result['score_range']}")
    
    return result

async def test_different_thumbnails_have_different_scores(test_images: List[bytes], niche: str) -> Dict[str, Any]:
    """Test that different thumbnails get different scores"""
    print(f"\n🔍 Testing score differentiation for {niche}...")
    
    # Create thumbnail list for comparison
    thumbnails = []
    for i, image_data in enumerate(test_images):
        thumbnails.append({
            "id": f"thumb_{i+1}",
            "image_data": image_data
        })
    
    # Score all thumbnails
    results = compare_thumbnails(thumbnails, niche)
    scores = [thumb["score"] for thumb in results["thumbnails"]]
    
    # Check if scores are different
    unique_scores = set(scores)
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    # For differentiation, we want at least 5 points difference between min and max
    differentiation_passed = score_range >= 5
    
    status = "✅ PASS" if differentiation_passed else "❌ FAIL"
    recommendation = "Good score differentiation" if differentiation_passed else f"Insufficient differentiation (range: {score_range})"
    
    print(f"   {status} - {recommendation}")
    print(f"   Scores: {scores}")
    print(f"   Range: {min_score}-{max_score} (spread: {score_range})")
    print(f"   Unique scores: {len(unique_scores)}/{len(scores)}")
    
    return {
        "test_passed": differentiation_passed,
        "scores": scores,
        "score_range": score_range,
        "unique_scores": len(unique_scores),
        "recommendation": recommendation
    }

async def test_all_niches(image_data: bytes, runs: int = 3) -> Dict[str, Any]:
    """Test consistency across all niches"""
    print(f"\n📊 Testing all niches for consistency...")
    
    niches = ["gaming", "business", "education", "tech", "food", "fitness", "entertainment", "travel", "music", "general"]
    
    results = {}
    for niche in niches:
        try:
            result = await test_single_niche_consistency(image_data, niche, runs)
            results[niche] = result
        except Exception as e:
            print(f"   ❌ FAIL - {niche}: {e}")
            results[niche] = {
                "test_passed": False,
                "error": str(e),
                "recommendation": f"Error testing {niche}"
            }
    
    # Summary
    passed = sum(1 for r in results.values() if r.get("test_passed", False))
    total = len(results)
    
    print(f"\n📈 Summary: {passed}/{total} niches passed consistency test")
    
    return {
        "passed_niches": passed,
        "total_niches": total,
        "niche_results": results,
        "overall_passed": passed == total
    }

async def test_api_endpoint(image_data: bytes, niche: str = "gaming") -> Dict[str, Any]:
    """Test the actual API endpoint"""
    print(f"\n🌐 Testing API endpoint with {niche} niche...")
    
    try:
        # Convert image to base64 data URL
        base64_image = base64.b64encode(image_data).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{base64_image}"
        
        # Prepare API request
        payload = {
            "title": "Test Thumbnail Scoring",
            "thumbnails": [
                {"id": "test1", "url": data_url}
            ],
            "category": niche
        }
        
        # Make API request
        response = requests.post(
            "http://localhost:8000/v1/score",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            score = result["thumbnails"][0]["ctr_score"]
            scoring_system = result["metadata"].get("scoring_system", "unknown")
            
            print(f"   ✅ API Response: {score}/100 (system: {scoring_system})")
            
            return {
                "test_passed": True,
                "score": score,
                "scoring_system": scoring_system,
                "response": result
            }
        else:
            print(f"   ❌ API Error: {response.status_code} - {response.text}")
            return {
                "test_passed": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    
    except Exception as e:
        print(f"   ❌ API Test Failed: {e}")
        return {
            "test_passed": False,
            "error": str(e)
        }

def create_different_test_images() -> List[bytes]:
    """Create different test images for differentiation testing"""
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    images = []
    
    # Image 1: High contrast, clear text
    img1 = Image.new('RGB', (1280, 720), color='#000000')
    draw1 = ImageDraw.Draw(img1)
    try:
        font = ImageFont.truetype("Arial.ttf", 120)
    except:
        font = ImageFont.load_default()
    draw1.text((300, 300), "HIGH CONTRAST", font=font, fill='white')
    
    buf1 = io.BytesIO()
    img1.save(buf1, format='JPEG', quality=95)
    images.append(buf1.getvalue())
    
    # Image 2: Low contrast, busy background
    img2 = Image.new('RGB', (1280, 720), color='#888888')
    draw2 = ImageDraw.Draw(img2)
    # Add noise
    for i in range(0, 1280, 20):
        for j in range(0, 720, 20):
            draw2.rectangle([i, j, i+10, j+10], fill='#999999')
    draw2.text((400, 350), "Low Contrast", font=font, fill='#777777')
    
    buf2 = io.BytesIO()
    img2.save(buf2, format='JPEG', quality=70)
    images.append(buf2.getvalue())
    
    # Image 3: Medium quality, colorful
    img3 = Image.new('RGB', (1280, 720), color='#2E86AB')
    draw3 = ImageDraw.Draw(img3)
    draw3.text((350, 300), "MEDIUM TEST", font=font, fill='#FFFFFF')
    
    buf3 = io.BytesIO()
    img3.save(buf3, format='JPEG', quality=85)
    images.append(buf3.getvalue())
    
    return images

async def main():
    parser = argparse.ArgumentParser(description='Test thumbnail scoring consistency')
    parser.add_argument('--niche', default='gaming', help='Niche to test (default: gaming)')
    parser.add_argument('--runs', type=int, default=5, help='Number of consistency runs (default: 5)')
    parser.add_argument('--test-image', help='Path to test image (optional)')
    parser.add_argument('--all-niches', action='store_true', help='Test all niches')
    parser.add_argument('--test-api', action='store_true', help='Test API endpoint')
    parser.add_argument('--test-differentiation', action='store_true', help='Test score differentiation')
    
    args = parser.parse_args()
    
    print("🧪 Thumbnail Scoring Consistency Test")
    print("=" * 50)
    
    # Load or create test image
    if args.test_image:
        print(f"📷 Loading test image: {args.test_image}")
        image_data = load_test_image(args.test_image)
    else:
        print("📷 Creating sample test image...")
        image_data = create_sample_test_image()
    
    print(f"✅ Image loaded: {len(image_data)} bytes")
    
    # Track overall results
    all_tests_passed = True
    test_results = {}
    
    # Test 1: Basic consistency for specified niche
    print(f"\n🔬 Test 1: Consistency for {args.niche} niche")
    consistency_result = await test_single_niche_consistency(image_data, args.niche, args.runs)
    test_results["consistency"] = consistency_result
    if not consistency_result["test_passed"]:
        all_tests_passed = False
    
    # Test 2: All niches (if requested)
    if args.all_niches:
        print(f"\n🔬 Test 2: All niches consistency")
        all_niches_result = await test_all_niches(image_data, runs=3)
        test_results["all_niches"] = all_niches_result
        if not all_niches_result["overall_passed"]:
            all_tests_passed = False
    
    # Test 3: Score differentiation (if requested)
    if args.test_differentiation:
        print(f"\n🔬 Test 3: Score differentiation")
        test_images = create_different_test_images()
        differentiation_result = await test_different_thumbnails_have_different_scores(test_images, args.niche)
        test_results["differentiation"] = differentiation_result
        if not differentiation_result["test_passed"]:
            all_tests_passed = False
    
    # Test 4: API endpoint (if requested)
    if args.test_api:
        print(f"\n🔬 Test 4: API endpoint")
        api_result = await test_api_endpoint(image_data, args.niche)
        test_results["api"] = api_result
        if not api_result["test_passed"]:
            all_tests_passed = False
    
    # Final summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Scoring system is consistent and reliable.")
    else:
        print("⚠️  SOME TESTS FAILED. Review the results above.")
    
    print(f"\n📊 Test Summary:")
    for test_name, result in test_results.items():
        status = "✅" if result.get("test_passed", False) else "❌"
        print(f"   {status} {test_name.title()}")
    
    print("\n💡 Recommendations:")
    if all_tests_passed:
        print("   - Scoring system is ready for production")
        print("   - Scores are consistent within ±2 points")
        print("   - Different thumbnails receive different scores")
    else:
        print("   - Review failed tests above")
        print("   - Check OpenAI API key and configuration")
        print("   - Verify network connectivity")
    
    # Return exit code
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)