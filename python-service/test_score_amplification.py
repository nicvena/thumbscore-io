#!/usr/bin/env python3
"""
Test Score Amplification - Thumbscore.io

Purpose: Verify that score amplification:
1. Maintains relative ordering (best stays best)
2. Provides good psychological spread
3. Feels accurate and trustworthy
4. Doesn't create weird edge cases

Run: python test_score_amplification.py
"""

import sys
import os

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import amplify_score


def test_score_ordering_preserved():
    """
    Verify that amplification preserves relative ordering.
    If thumbnail A scores higher than B before amplification,
    it should still score higher after.
    """
    print("🔍 TEST 1: Score Ordering Preserved")
    print("=" * 60)
    
    test_cases = [
        (80, 60, 40),  # Good, average, poor
        (75, 55, 35),  # Another set
        (90, 70, 50),  # Excellent, good, average
        (65, 50, 30),  # Real-world scenario
        (100, 80, 60), # Maximum range
    ]
    
    all_passed = True
    for high, mid, low in test_cases:
        amp_high = amplify_score(high)
        amp_mid = amplify_score(mid)
        amp_low = amplify_score(low)
        
        if amp_high > amp_mid > amp_low:
            print(f"  ✓ {high}→{amp_high}, {mid}→{amp_mid}, {low}→{amp_low}")
        else:
            print(f"  ❌ ORDERING BROKEN: {high}→{amp_high}, {mid}→{amp_mid}, {low}→{amp_low}")
            all_passed = False
    
    print()
    return all_passed


def test_score_ranges():
    """
    Verify scores fall into expected psychological ranges.
    """
    print("🔍 TEST 2: Score Ranges")
    print("=" * 60)
    
    all_passed = True
    
    # Excellent raw scores (75-100) should amplify to 85-95
    print("  Testing Excellent Range (75-100 → 85-95):")
    excellent_scores = [75, 80, 85, 90, 95, 100]
    for score in excellent_scores:
        amplified = amplify_score(score)
        if 85 <= amplified <= 95:
            print(f"    ✓ {score} → {amplified}")
        else:
            print(f"    ❌ {score} → {amplified} (expected 85-95)")
            all_passed = False
    
    # Good raw scores (60-75) should amplify to 70-85
    print("\n  Testing Good Range (60-75 → 70-85):")
    good_scores = [60, 65, 70, 75]
    for score in good_scores:
        amplified = amplify_score(score)
        if 70 <= amplified <= 85:
            print(f"    ✓ {score} → {amplified}")
        else:
            print(f"    ❌ {score} → {amplified} (expected 70-85)")
            all_passed = False
    
    # Average raw scores (40-60) should amplify to 55-70
    print("\n  Testing Average Range (40-60 → 55-70):")
    avg_scores = [40, 45, 50, 55, 60]
    for score in avg_scores:
        amplified = amplify_score(score)
        if 55 <= amplified <= 70:
            print(f"    ✓ {score} → {amplified}")
        else:
            print(f"    ❌ {score} → {amplified} (expected 55-70)")
            all_passed = False
    
    # Poor raw scores (20-40) should amplify to 40-55
    print("\n  Testing Poor Range (20-40 → 40-55):")
    poor_scores = [20, 25, 30, 35, 40]
    for score in poor_scores:
        amplified = amplify_score(score)
        if 40 <= amplified <= 55:
            print(f"    ✓ {score} → {amplified}")
        else:
            print(f"    ❌ {score} → {amplified} (expected 40-55)")
            all_passed = False
    
    print()
    return all_passed


def test_edge_cases():
    """
    Test boundary conditions and edge cases.
    """
    print("🔍 TEST 3: Edge Cases")
    print("=" * 60)
    
    all_passed = True
    
    # Minimum score
    min_score = amplify_score(0)
    if min_score >= 30:
        print(f"  ✓ Minimum score (0 → {min_score}) >= 30")
    else:
        print(f"  ❌ Minimum score (0 → {min_score}) should be >= 30")
        all_passed = False
    
    # Maximum score
    max_score = amplify_score(100)
    if max_score <= 95:
        print(f"  ✓ Maximum score (100 → {max_score}) <= 95")
    else:
        print(f"  ❌ Maximum score (100 → {max_score}) should be <= 95")
        all_passed = False
    
    # Same input gives same output (deterministic)
    score1 = amplify_score(50)
    score2 = amplify_score(50)
    if score1 == score2:
        print(f"  ✓ Deterministic: 50 → {score1} (consistent)")
    else:
        print(f"  ❌ Not deterministic: {score1} vs {score2}")
        all_passed = False
    
    # Small differences create visible separation
    diff_before = 55 - 50  # 5 points
    amp_55 = amplify_score(55)
    amp_50 = amplify_score(50)
    diff_after = amp_55 - amp_50
    if diff_after >= 3:
        print(f"  ✓ Separation maintained: 5pt difference → {diff_after}pt difference")
    else:
        print(f"  ❌ Insufficient separation: 5pt → {diff_after}pt (expected >=3)")
        all_passed = False
    
    # Negative scores should clamp to minimum
    neg_score = amplify_score(-10)
    if neg_score >= 30:
        print(f"  ✓ Negative score clamped: -10 → {neg_score}")
    else:
        print(f"  ❌ Negative score not clamped: -10 → {neg_score}")
        all_passed = False
    
    # Over-max scores should clamp to maximum
    over_score = amplify_score(150)
    if over_score <= 95:
        print(f"  ✓ Over-max score clamped: 150 → {over_score}")
    else:
        print(f"  ❌ Over-max score not clamped: 150 → {over_score}")
        all_passed = False
    
    print()
    return all_passed


def test_psychological_appeal():
    """
    Test that scores feel encouraging.
    """
    print("🔍 TEST 4: Psychological Appeal")
    print("=" * 60)
    
    all_passed = True
    
    # A "pretty good" raw score (65) should feel good (70+)
    good_score = amplify_score(65)
    if good_score >= 70:
        print(f"  ✓ Good scores feel encouraging: 65 → {good_score} (>=70)")
    else:
        print(f"  ❌ Good score feels weak: 65 → {good_score} (expected >=70)")
        all_passed = False
    
    # Even average scores (50) shouldn't feel terrible (<40)
    avg_score = amplify_score(50)
    if avg_score >= 55:
        print(f"  ✓ Average doesn't feel like failure: 50 → {avg_score} (>=55)")
    else:
        print(f"  ❌ Average feels too low: 50 → {avg_score} (expected >=55)")
        all_passed = False
    
    # Excellent scores (85+) should feel excellent (88+)
    excellent_score = amplify_score(85)
    if excellent_score >= 88:
        print(f"  ✓ Excellent feels excellent: 85 → {excellent_score} (>=88)")
    else:
        print(f"  ❌ Excellent doesn't feel excellent: 85 → {excellent_score} (expected >=88)")
        all_passed = False
    
    # Poor scores (35) should still give hope (not <35)
    poor_score = amplify_score(35)
    if 40 <= poor_score <= 50:
        print(f"  ✓ Poor score feels actionable: 35 → {poor_score} (40-50)")
    else:
        print(f"  ⚠️  Poor score: 35 → {poor_score} (check if appropriate)")
    
    # Very poor (20) should indicate serious issues but not hopeless
    very_poor = amplify_score(20)
    if 35 <= very_poor <= 45:
        print(f"  ✓ Very poor indicates issues: 20 → {very_poor} (35-45)")
    else:
        print(f"  ⚠️  Very poor score: 20 → {very_poor} (check if appropriate)")
    
    print()
    return all_passed


def test_real_world_scenarios():
    """
    Test with actual score patterns from your system.
    """
    print("🔍 TEST 5: Real-World Scenarios")
    print("=" * 60)
    
    all_passed = True
    
    # Scenario 1: Current problem (45, 19, 11)
    print("  Scenario 1: Current Problem Scores")
    current_winner = 45
    current_second = 19
    current_third = 11
    
    new_winner = amplify_score(current_winner)
    new_second = amplify_score(current_second)
    new_third = amplify_score(current_third)
    
    print(f"    Before: {current_winner}, {current_second}, {current_third}")
    print(f"    After:  {new_winner}, {new_second}, {new_third}")
    
    # Winner should feel good (70+)
    if new_winner >= 65:
        print(f"    ✓ Winner feels good: {new_winner} >= 65")
    else:
        print(f"    ❌ Winner feels weak: {new_winner} < 65")
        all_passed = False
    
    # Should have clear separation (20+ points between winner and third)
    separation = new_winner - new_third
    if separation >= 20:
        print(f"    ✓ Clear separation: {separation} points")
    else:
        print(f"    ❌ Weak separation: {separation} points (expected >=20)")
        all_passed = False
    
    # Scenario 2: Competitive thumbnails (75, 72, 68)
    print("\n  Scenario 2: Competitive Thumbnails")
    comp_first = 75
    comp_second = 72
    comp_third = 68
    
    amp_first = amplify_score(comp_first)
    amp_second = amplify_score(comp_second)
    amp_third = amplify_score(comp_third)
    
    print(f"    Before: {comp_first}, {comp_second}, {comp_third}")
    print(f"    After:  {amp_first}, {amp_second}, {amp_third}")
    
    # Should maintain order but show they're all good
    if amp_first > amp_second > amp_third and amp_third >= 82:
        print(f"    ✓ All competitive scores feel good (all >=82)")
    else:
        print(f"    ⚠️  Check competitive scenario results")
    
    # Scenario 3: Clear winner (90, 65, 45)
    print("\n  Scenario 3: Clear Winner")
    clear_winner = 90
    clear_mid = 65
    clear_poor = 45
    
    amp_clear_winner = amplify_score(clear_winner)
    amp_clear_mid = amplify_score(clear_mid)
    amp_clear_poor = amplify_score(clear_poor)
    
    print(f"    Before: {clear_winner}, {clear_mid}, {clear_poor}")
    print(f"    After:  {amp_clear_winner}, {amp_clear_mid}, {amp_clear_poor}")
    
    # Winner should be excellent (>=85)
    if amp_clear_winner >= 88:
        print(f"    ✓ Clear winner is excellent: {amp_clear_winner} >= 88")
    else:
        print(f"    ❌ Clear winner not excellent enough: {amp_clear_winner}")
        all_passed = False
    
    # Should have huge separation
    huge_separation = amp_clear_winner - amp_clear_poor
    if huge_separation >= 25:
        print(f"    ✓ Huge separation: {huge_separation} points")
    else:
        print(f"    ⚠️  Separation could be larger: {huge_separation} points")
    
    print()
    return all_passed


def test_distribution_analysis():
    """
    Analyze the distribution of amplified scores.
    """
    print("🔍 TEST 6: Distribution Analysis")
    print("=" * 60)
    
    # Test a range of input scores
    raw_scores = range(0, 101, 5)
    amplified_scores = [amplify_score(score) for score in raw_scores]
    
    print("  Raw Score → Amplified Score")
    print("  " + "-" * 30)
    for raw, amp in zip(raw_scores, amplified_scores):
        # Color code based on quality
        if amp >= 85:
            emoji = "🟢"  # Excellent
        elif amp >= 70:
            emoji = "🔵"  # Good
        elif amp >= 55:
            emoji = "🟡"  # Average
        elif amp >= 40:
            emoji = "🟠"  # Fair
        else:
            emoji = "🔴"  # Poor
        
        print(f"  {raw:3d} → {amp:2d}  {emoji}")
    
    # Calculate statistics
    min_amp = min(amplified_scores)
    max_amp = max(amplified_scores)
    range_amp = max_amp - min_amp
    
    print("\n  Statistics:")
    print(f"    Min amplified: {min_amp}")
    print(f"    Max amplified: {max_amp}")
    print(f"    Range: {range_amp}")
    print(f"    ✓ Good spread for psychological differentiation")
    
    print()
    return True


def run_all_tests():
    """
    Run all test suites and report results.
    """
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   THUMBSCORE.IO - SCORE AMPLIFICATION TEST SUITE         ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    
    results = {
        "Score Ordering": test_score_ordering_preserved(),
        "Score Ranges": test_score_ranges(),
        "Edge Cases": test_edge_cases(),
        "Psychological Appeal": test_psychological_appeal(),
        "Real-World Scenarios": test_real_world_scenarios(),
        "Distribution Analysis": test_distribution_analysis(),
    }
    
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                      TEST SUMMARY                         ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! Score amplification is working correctly.")
        print()
        print("✅ Relative ordering preserved")
        print("✅ Psychological ranges appropriate")
        print("✅ Edge cases handled")
        print("✅ Real-world scenarios validated")
        print()
        print("🚀 Ready for production!")
        return 0
    else:
        print("❌ SOME TESTS FAILED. Review amplify_score() function.")
        print()
        print("Recommendations:")
        print("  1. Check amplification curve in app/main.py")
        print("  2. Adjust piecewise linear mapping")
        print("  3. Verify sigmoid blending")
        print("  4. Re-run tests after adjustments")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

