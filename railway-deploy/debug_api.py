#!/usr/bin/env python3
"""
Debug the actual API response to see what's happening
"""

import requests
import json

def test_api_directly():
    """Test the API directly to see the response"""
    
    # Test data similar to what the frontend sends
    test_data = {
        "thumbnails": [
            {
                "id": "test1",
                "url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
                "title": "Test Video"
            }
        ]
    }
    
    try:
        print("Testing API directly...")
        response = requests.post(
            "http://localhost:8000/v1/score",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response JSON: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Raw Response: {response.text[:500]}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api_directly()
