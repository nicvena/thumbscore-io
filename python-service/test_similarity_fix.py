#!/usr/bin/env python3
"""
Isolated test to verify similarity field is handled correctly
"""
import sys
sys.path.insert(0, '/Users/nicvenettacci/Desktop/Thumbnail Lab/thumbnail-lab/python-service')

from app.main import SubScores, explain, ThumbnailScore, Overlays

print("Testing SubScores with all fields...")
try:
    subscores_dict = {
        'similarity': 82,
        'power_words': 69,
        'brain_weighted': 82,
        'clarity': 82,
        'subject_prominence': 92,
        'contrast_pop': 82,
        'emotion': 64,
        'hierarchy': 82,
        'title_match': 82
    }
    
    subscores_obj = SubScores(**subscores_dict)
    print(f"✓ SubScores created successfully: {subscores_obj}")
    
    # Test explain function
    overlays = Overlays(
        saliency_heatmap_url="/test/heatmap.png",
        ocr_boxes_url="/test/ocr.png",
        face_boxes_url="/test/faces.png"
    )
    
    result = ThumbnailScore(
        id="test1",
        ctr_score=78.5,
        subscores=subscores_obj,
        insights=["Test insight"],
        overlays=overlays
    )
    
    print(f"✓ ThumbnailScore created successfully")
    
    # Test explain function
    explanation = explain([result], "test1")
    print(f"✓ Explanation generated: {explanation}")
    
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

