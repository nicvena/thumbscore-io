"""
Niche-specific insights and recommendations for thumbnail analysis results

This module provides tailored coaching and recommendations based on the user's
content niche and their thumbnail's component scores.
"""

from typing import List, Dict, Any

# Niche-specific insights database
NICHE_INSIGHTS = {
    'gaming': {
        'high_emotion_tip': '🎮 Gaming thumbnails with exaggerated expressions get 45% more clicks. Your emotion score is excellent!',
        'low_emotion_tip': '⚠️ Gaming audiences respond to shocked/excited faces. Consider adding more expressive reactions.',
        'high_color_tip': '✨ Bright, saturated colors are perfect for gaming content. Your color pop score is crushing it!',
        'low_color_tip': '🎨 Gaming thumbnails perform better with high saturation and contrast. Try boosting your colors.',
        'power_words_tip': '💥 Words like "INSANE", "BROKEN", "OP" drive massive engagement in gaming.',
        'general_tip': '🏆 Top gaming thumbnails average 88/100. You\'re in the top tier!',
        'text_placement': 'Gaming: Place text at top or bottom with bold outlines for maximum impact.',
        'baseline_score': 78
    },
    'business': {
        'high_clarity_tip': '💼 Professional, clear text is critical in business content. Excellent clarity score!',
        'low_clarity_tip': '📝 Business audiences want clear, readable information. Simplify your text.',
        'high_composition_tip': '✅ Clean, professional layout builds trust. Your composition is perfect.',
        'low_composition_tip': '⚠️ Business thumbnails need clean, professional layouts. Reduce clutter.',
        'power_words_tip': '📈 Words like "PROVEN", "STRATEGY", "FRAMEWORK" establish credibility.',
        'general_tip': '💡 Business thumbnails should look professional, not flashy. You nailed it!',
        'text_placement': 'Business: Center or left-aligned text with professional fonts work best.',
        'baseline_score': 75
    },
    'education': {
        'high_clarity_tip': '🎓 Educational content needs crystal-clear text. Your clarity is excellent!',
        'low_clarity_tip': '📚 Learners need to understand your topic instantly. Simplify your text.',
        'high_composition_tip': '✅ Well-organized thumbnails signal quality education. Great job!',
        'low_composition_tip': '⚠️ Educational thumbnails should look structured and clear. Reduce visual noise.',
        'power_words_tip': '📖 Words like "LEARN", "MASTER", "COMPLETE" attract students.',
        'general_tip': '🌟 Top educational thumbnails are clear, organized, and promise value.',
        'text_placement': 'Education: Clear, hierarchical text helps learners know what to expect.',
        'baseline_score': 72
    },
    'tech': {
        'high_composition_tip': '💻 Clean, modern layouts work best in tech. Perfect composition!',
        'low_composition_tip': '🔧 Tech audiences appreciate minimalist, product-focused designs.',
        'high_clarity_tip': '✅ Clear product/feature communication is key. Excellent clarity!',
        'low_clarity_tip': '⚠️ Tech viewers want to see the product clearly. Reduce distractions.',
        'power_words_tip': '🔍 Words like "REVIEW", "VS", "WORTH IT" drive tech content clicks.',
        'general_tip': '⚡ Tech thumbnails should be clean, modern, and product-focused.',
        'text_placement': 'Tech: Integrate text with product images for a cohesive look.',
        'baseline_score': 74
    },
    'food': {
        'high_color_tip': '🍳 Food thumbnails live or die by color! Your vibrant colors are perfect!',
        'low_color_tip': '🎨 Make your food look delicious with warm, appetizing colors.',
        'high_composition_tip': '✅ Close-up food shots with great lighting = clicks. Nailed it!',
        'low_composition_tip': '📸 Get closer! Food thumbnails need mouth-watering close-ups.',
        'power_words_tip': '😋 Words like "EASY", "QUICK", "PERFECT" make recipes irresistible.',
        'general_tip': '🌟 Food thumbnails are all about making viewers hungry. Mission accomplished!',
        'text_placement': 'Food: Minimal text, let the food be the star. Add time/difficulty only.',
        'baseline_score': 76
    },
    'fitness': {
        'high_subject_tip': '💪 Clear focus on body/transformation is essential. Perfect subject prominence!',
        'low_subject_tip': '🏋️ Show the transformation/body clearly. This is what viewers want to see.',
        'high_composition_tip': '✅ Strong before/after or action shots = engagement. Great work!',
        'low_composition_tip': '📸 Fitness thumbnails need clear subject focus. Remove background distractions.',
        'power_words_tip': '🔥 Words like "TRANSFORMATION", "DAYS", "FAST" drive fitness clicks.',
        'general_tip': '💯 Fitness thumbnails sell results. Show the transformation prominently!',
        'text_placement': 'Fitness: Add text showing timeframe (30 Days) or result (Lost 20lbs).',
        'baseline_score': 73
    },
    'entertainment': {
        'high_emotion_tip': '🎬 Entertainment thrives on emotion! Your expressive thumbnail is perfect!',
        'low_emotion_tip': '😮 Entertainment needs personality. Show more expression!',
        'high_color_tip': '✨ Bold, eye-catching colors grab attention. Crushing it!',
        'low_color_tip': '🎨 Entertainment can be more vibrant. Don\'t be afraid of bold colors!',
        'power_words_tip': '🔥 Words like "SHOCKING", "REVEALED", "SECRET" create curiosity.',
        'general_tip': '🌟 Entertainment is all about personality and emotion. Show yourself!',
        'text_placement': 'Entertainment: Big, bold text that creates curiosity works best.',
        'baseline_score': 77
    },
    'music': {
        'high_composition_tip': '🎵 Artistic, visually striking thumbnails = music clicks. Beautiful!',
        'low_composition_tip': '🎨 Music thumbnails can be more artistic and visually bold.',
        'high_color_tip': '✅ Music allows for creative color choices. Your palette is perfect!',
        'low_color_tip': '🌈 Music gives you freedom to be creative with colors. Try bolder choices!',
        'power_words_tip': '🎧 Words like "OFFICIAL", "MUSIC VIDEO", "LIVE" establish legitimacy.',
        'general_tip': '🎤 Music thumbnails blend artistry with branding. Great balance!',
        'text_placement': 'Music: Artist name + song title, keep it simple and readable.',
        'baseline_score': 79
    },
    'travel': {
        'high_composition_tip': '✈️ Stunning landscapes make travel thumbnails irresistible. Perfect composition!',
        'low_composition_tip': '🏔️ Travel thumbnails need breathtaking visuals. Show the destination clearly.',
        'high_color_tip': '🌅 Vibrant, aspirational colors drive travel clicks. Your palette is perfect!',
        'low_color_tip': '🎨 Travel content needs vibrant, eye-catching colors to inspire wanderlust.',
        'high_clarity_tip': '✅ Clear location names help viewers decide if they want to watch. Great clarity!',
        'low_clarity_tip': '📍 Make your destination/location crystal clear in the text.',
        'power_words_tip': '🗺️ Words like "HIDDEN", "SECRET", "PARADISE" create curiosity about destinations.',
        'general_tip': '🌍 Travel thumbnails should make viewers say "I want to go there!"',
        'text_placement': 'Travel: Feature location name prominently, keep text minimal to showcase scenery.',
        'baseline_score': 76
    },
    'general': {
        'balanced_tip': '📺 Your thumbnail shows balanced scoring across categories.',
        'improve_tip': '💡 Focus on your weakest scoring component for biggest improvement.',
        'power_words_tip': '📝 Strong power words increase clicks by up to 340%.',
        'general_tip': '✨ General content benefits from clear messaging and good composition.',
        'text_placement': 'General: Keep text simple, clear, and easy to read at small sizes.',
        'baseline_score': 70
    }
}

def get_niche_insights(niche_id: str, components: Dict[str, float]) -> List[str]:
    """
    Get relevant insights based on niche and component scores
    
    Args:
        niche_id (str): The content niche ID ('gaming', 'business', etc.)
        components (dict): Component scores (0-100 scale)
            - emotion: Emotional expression score
            - color_pop: Color saturation/contrast score  
            - text_clarity: Text readability score
            - composition: Visual composition score
            - subject_prominence: Main subject focus score
            - similarity: Similarity to successful thumbnails
    
    Returns:
        List[str]: List of relevant insights (max 4)
    """
    insights = []
    niche_tips = NICHE_INSIGHTS.get(niche_id, NICHE_INSIGHTS['general'])
    
    # Add score-specific insights based on component performance
    
    # Emotion scoring
    emotion_score = components.get('emotion', 0)
    if emotion_score > 80 and 'high_emotion_tip' in niche_tips:
        insights.append(niche_tips['high_emotion_tip'])
    elif emotion_score < 40 and 'low_emotion_tip' in niche_tips:
        insights.append(niche_tips['low_emotion_tip'])
    
    # Color pop scoring
    color_score = components.get('color_pop', 0)
    if color_score > 80 and 'high_color_tip' in niche_tips:
        insights.append(niche_tips['high_color_tip'])
    elif color_score < 40 and 'low_color_tip' in niche_tips:
        insights.append(niche_tips['low_color_tip'])
    
    # Text clarity scoring
    clarity_score = components.get('text_clarity', 0)
    if clarity_score > 80 and 'high_clarity_tip' in niche_tips:
        insights.append(niche_tips['high_clarity_tip'])
    elif clarity_score < 40 and 'low_clarity_tip' in niche_tips:
        insights.append(niche_tips['low_clarity_tip'])
    
    # Composition scoring
    composition_score = components.get('composition', 0)
    if composition_score > 80 and 'high_composition_tip' in niche_tips:
        insights.append(niche_tips['high_composition_tip'])
    elif composition_score < 40 and 'low_composition_tip' in niche_tips:
        insights.append(niche_tips['low_composition_tip'])
    
    # Subject prominence scoring
    subject_score = components.get('subject_prominence', 0)
    if subject_score > 80 and 'high_subject_tip' in niche_tips:
        insights.append(niche_tips['high_subject_tip'])
    elif subject_score < 40 and 'low_subject_tip' in niche_tips:
        insights.append(niche_tips['low_subject_tip'])
    
    # Always add power words and general tips
    if 'power_words_tip' in niche_tips:
        insights.append(niche_tips['power_words_tip'])
    
    if 'general_tip' in niche_tips:
        insights.append(niche_tips['general_tip'])
    
    # Add text placement advice
    if 'text_placement' in niche_tips:
        insights.append(niche_tips['text_placement'])
    
    # Return top 4 most relevant insights
    return insights[:4]

def get_niche_baseline(niche_id: str) -> int:
    """Get baseline score for niche calibration"""
    niche_config = NICHE_INSIGHTS.get(niche_id, NICHE_INSIGHTS['general'])
    return niche_config.get('baseline_score', 70)

def validate_niche_id(niche_id: str) -> bool:
    """Check if niche_id is valid"""
    return niche_id in NICHE_INSIGHTS

def get_available_niches() -> List[Dict[str, Any]]:
    """Get list of all available niches for frontend"""
    niches = []
    niche_info = {
        'gaming': {'name': 'Gaming', 'emoji': '🎮', 'description': 'Video games, streaming, esports'},
        'business': {'name': 'Business & Finance', 'emoji': '💼', 'description': 'Entrepreneurship, investing, career'},
        'education': {'name': 'Education & How-To', 'emoji': '🎓', 'description': 'Tutorials, courses, educational content'},
        'tech': {'name': 'Tech & Reviews', 'emoji': '💻', 'description': 'Technology reviews, gadgets, software'},
        'food': {'name': 'Food & Cooking', 'emoji': '🍳', 'description': 'Recipes, cooking tutorials, food reviews'},
        'fitness': {'name': 'Fitness & Health', 'emoji': '💪', 'description': 'Workouts, fitness tips, nutrition'},
        'entertainment': {'name': 'Entertainment & Vlogs', 'emoji': '🎬', 'description': 'Vlogs, entertainment, comedy'},
        'travel': {'name': 'Travel & Lifestyle', 'emoji': '✈️', 'description': 'Travel vlogs, destination guides, adventure content'},
        'music': {'name': 'Music', 'emoji': '🎵', 'description': 'Music videos, covers, tutorials'},
        'general': {'name': 'General / Other', 'emoji': '📺', 'description': 'General content or mixed categories'}
    }
    
    for niche_id in NICHE_INSIGHTS.keys():
        info = niche_info.get(niche_id, {})
        niches.append({
            'id': niche_id,
            'name': info.get('name', niche_id.title()),
            'emoji': info.get('emoji', '📺'),
            'description': info.get('description', 'Content category'),
            'baseline_score': get_niche_baseline(niche_id)
        })
    
    return niches

# Example usage and testing
if __name__ == "__main__":
    # Test gaming niche with high emotion, low clarity
    test_components = {
        'emotion': 85,
        'color_pop': 75,
        'text_clarity': 35,
        'composition': 70,
        'subject_prominence': 60,
        'similarity': 80
    }
    
    insights = get_niche_insights('gaming', test_components)
    print("Gaming Niche Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print(f"\nGaming baseline score: {get_niche_baseline('gaming')}")
    print(f"Is 'gaming' valid niche? {validate_niche_id('gaming')}")