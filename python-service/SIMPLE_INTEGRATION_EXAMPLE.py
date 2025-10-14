"""
SIMPLE INTEGRATION EXAMPLE

Add these lines to your existing analyze endpoint to get niche insights working:
"""

# 1. Import the niche insights module
from niche_insights import get_niche_insights, validate_niche_id, get_niche_baseline

@app.post("/api/analyze")
async def analyze_thumbnails(request):
    """Your existing analyze endpoint"""
    
    # 2. Extract and validate niche
    niche_id = request.get('niche', 'general')
    if not validate_niche_id(niche_id):
        niche_id = 'general'
    
    # 3. Your existing scoring logic here...
    # thumbnails = process_thumbnails(request.thumbnails)
    # scores = calculate_scores(thumbnails)
    
    results = []
    for i, thumbnail in enumerate(request.thumbnails):
        # Your existing component scoring
        components = {
            'emotion': 75.0,           # Your emotion detection result
            'color_pop': 82.0,         # Your color analysis result  
            'text_clarity': 68.0,      # Your OCR/clarity analysis
            'composition': 79.0,       # Your composition analysis
            'subject_prominence': 71.0, # Your subject detection
            'similarity': 85.0         # Your FAISS similarity score
        }
        
        # Calculate overall score with your existing logic
        overall_score = calculate_overall_score(components)
        
        # 4. Get niche-specific insights - ADD THIS LINE
        insights = get_niche_insights(niche_id, components)
        
        # Add to results
        results.append({
            'id': i + 1,
            'score': overall_score,
            'components': components,
            'niche_insights': insights,  # <- NEW: Add insights to response
            # ... your other existing fields
        })
    
    # Find winner
    winner_index = max(range(len(results)), key=lambda i: results[i]['score'])
    winner_score = results[winner_index]['score']
    
    # 5. Return response with niche data - MODIFY YOUR RETURN
    return {
        'overall_score': winner_score,
        'niche_used': niche_id,                                    # <- NEW
        'niche_insights': results[winner_index]['niche_insights'], # <- NEW
        'thumbnails': results,
        # ... your existing response fields
    }

# That's it! Your frontend will now receive niche insights.

# ============================================================
# FRONTEND USAGE (after backend integration above):
# ============================================================

"""
// In your React results component:

useEffect(() => {
  const fetchAnalysis = async () => {
    const niche = sessionStorage.getItem('selectedNiche') || 'general';
    
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        thumbnails: thumbnails,
        title: title,
        niche: niche  // <- Pass niche to backend
      })
    });

    const data = await response.json();
    
    // Now data contains:
    // - data.niche_used
    // - data.niche_insights  
    // - data.thumbnails[].niche_insights
    
    setAnalysisResult(data);
  };
  
  fetchAnalysis();
}, []);

// Display in your JSX:
<NicheBadge niche={analysisResult.niche_used} />
<NicheInsights insights={analysisResult.niche_insights} />
"""

# ============================================================
# TESTING THE INTEGRATION:
# ============================================================

def test_niche_insights():
    """Test the niche insights system"""
    
    # Test gaming niche with high emotion
    gaming_components = {
        'emotion': 88.0,           # High emotion (good for gaming)
        'color_pop': 75.0,         # Decent color
        'text_clarity': 45.0,      # Low clarity (needs work)
        'composition': 70.0,       # OK composition  
        'subject_prominence': 65.0, # OK subject focus
        'similarity': 82.0         # Good similarity
    }
    
    insights = get_niche_insights('gaming', gaming_components)
    print("Gaming Insights:")
    for insight in insights:
        print(f"  - {insight}")
    
    # Test business niche with high clarity
    business_components = {
        'emotion': 35.0,           # Low emotion (OK for business)
        'color_pop': 55.0,         # Moderate color (OK for business)
        'text_clarity': 89.0,      # High clarity (great for business)
        'composition': 85.0,       # High composition (great for business)
        'subject_prominence': 60.0, # OK subject focus
        'similarity': 78.0         # Good similarity
    }
    
    insights = get_niche_insights('business', business_components)
    print("\nBusiness Insights:")
    for insight in insights:
        print(f"  - {insight}")

if __name__ == "__main__":
    test_niche_insights()