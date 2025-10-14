#!/usr/bin/env python3
"""
Simple test to isolate the timeout issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.power_words import score_power_words

def test_power_words():
    """Test power words scoring in isolation"""
    
    print("Testing power words scoring...")
    
    # Test with empty text (should return 30)
    result1 = score_power_words("", "tech")
    print(f"Empty text result: {result1}")
    
    # Test with some text
    result2 = score_power_words("INSANE iPhone TEST", "tech")
    print(f"Normal text result: {result2}")
    
    # Test with None
    result3 = score_power_words(None, "tech")
    print(f"None text result: {result3}")

if __name__ == "__main__":
    test_power_words()
