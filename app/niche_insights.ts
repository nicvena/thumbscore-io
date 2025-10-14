/**
 * Niche-specific insights and recommendations for thumbnail analysis results
 */

interface ComponentScores {
  emotion?: number;
  color_pop?: number;
  text_clarity?: number;
  composition?: number;
  subject_prominence?: number;
  similarity?: number;
}

interface NicheTips {
  high_emotion_tip?: string;
  low_emotion_tip?: string;
  high_color_tip?: string;
  low_color_tip?: string;
  high_clarity_tip?: string;
  low_clarity_tip?: string;
  high_composition_tip?: string;
  low_composition_tip?: string;
  high_subject_tip?: string;
  low_subject_tip?: string;
  power_words_tip?: string;
  general_tip?: string;
  text_placement?: string;
  balanced_tip?: string;
  improve_tip?: string;
}

const NICHE_INSIGHTS: { [key: string]: NicheTips } = {
  'gaming': {
    'high_emotion_tip': '🎮 Gaming thumbnails with exaggerated expressions get 45% more clicks. Your emotion score is excellent!',
    'low_emotion_tip': '⚠️ Gaming audiences respond to shocked/excited faces. Consider adding more expressive reactions.',
    'high_color_tip': '✨ Bright, saturated colors are perfect for gaming content. Your color pop score is crushing it!',
    'low_color_tip': '🎨 Gaming thumbnails perform better with high saturation and contrast. Try boosting your colors.',
    'power_words_tip': '💥 Words like "INSANE", "BROKEN", "OP" drive massive engagement in gaming.',
    'general_tip': '🏆 Top gaming thumbnails average 88/100. You\'re in the top tier!',
    'text_placement': 'Gaming: Place text at top or bottom with bold outlines for maximum impact.'
  },
  'business': {
    'high_clarity_tip': '💼 Professional, clear text is critical in business content. Excellent clarity score!',
    'low_clarity_tip': '📝 Business audiences want clear, readable information. Simplify your text.',
    'high_composition_tip': '✅ Clean, professional layout builds trust. Your composition is perfect.',
    'low_composition_tip': '⚠️ Business thumbnails need clean, professional layouts. Reduce clutter.',
    'power_words_tip': '📈 Words like "PROVEN", "STRATEGY", "FRAMEWORK" establish credibility.',
    'general_tip': '💡 Business thumbnails should look professional, not flashy. You nailed it!',
    'text_placement': 'Business: Center or left-aligned text with professional fonts work best.'
  },
  'education': {
    'high_clarity_tip': '🎓 Educational content needs crystal-clear text. Your clarity is excellent!',
    'low_clarity_tip': '📚 Learners need to understand your topic instantly. Simplify your text.',
    'high_composition_tip': '✅ Well-organized thumbnails signal quality education. Great job!',
    'low_composition_tip': '⚠️ Educational thumbnails should look structured and clear. Reduce visual noise.',
    'power_words_tip': '📖 Words like "LEARN", "MASTER", "COMPLETE" attract students.',
    'general_tip': '🌟 Top educational thumbnails are clear, organized, and promise value.',
    'text_placement': 'Education: Clear, hierarchical text helps learners know what to expect.'
  },
  'tech': {
    'high_composition_tip': '💻 Clean, modern layouts work best in tech. Perfect composition!',
    'low_composition_tip': '🔧 Tech audiences appreciate minimalist, product-focused designs.',
    'high_clarity_tip': '✅ Clear product/feature communication is key. Excellent clarity!',
    'low_clarity_tip': '⚠️ Tech viewers want to see the product clearly. Reduce distractions.',
    'power_words_tip': '🔍 Words like "REVIEW", "VS", "WORTH IT" drive tech content clicks.',
    'general_tip': '⚡ Tech thumbnails should be clean, modern, and product-focused.',
    'text_placement': 'Tech: Integrate text with product images for a cohesive look.'
  },
  'food': {
    'high_color_tip': '🍳 Food thumbnails live or die by color! Your vibrant colors are perfect!',
    'low_color_tip': '🎨 Make your food look delicious with warm, appetizing colors.',
    'high_composition_tip': '✅ Close-up food shots with great lighting = clicks. Nailed it!',
    'low_composition_tip': '📸 Get closer! Food thumbnails need mouth-watering close-ups.',
    'power_words_tip': '😋 Words like "EASY", "QUICK", "PERFECT" make recipes irresistible.',
    'general_tip': '🌟 Food thumbnails are all about making viewers hungry. Mission accomplished!',
    'text_placement': 'Food: Minimal text, let the food be the star. Add time/difficulty only.'
  },
  'fitness': {
    'high_subject_tip': '💪 Clear focus on body/transformation is essential. Perfect subject prominence!',
    'low_subject_tip': '🏋️ Show the transformation/body clearly. This is what viewers want to see.',
    'high_composition_tip': '✅ Strong before/after or action shots = engagement. Great work!',
    'low_composition_tip': '📸 Fitness thumbnails need clear subject focus. Remove background distractions.',
    'power_words_tip': '🔥 Words like "TRANSFORMATION", "DAYS", "FAST" drive fitness clicks.',
    'general_tip': '💯 Fitness thumbnails sell results. Show the transformation prominently!',
    'text_placement': 'Fitness: Add text showing timeframe (30 Days) or result (Lost 20lbs).'
  },
  'entertainment': {
    'high_emotion_tip': '🎬 Entertainment thrives on emotion! Your expressive thumbnail is perfect!',
    'low_emotion_tip': '😮 Entertainment needs personality. Show more expression!',
    'high_color_tip': '✨ Bold, eye-catching colors grab attention. Crushing it!',
    'low_color_tip': '🎨 Entertainment can be more vibrant. Don\'t be afraid of bold colors!',
    'power_words_tip': '🔥 Words like "SHOCKING", "REVEALED", "SECRET" create curiosity.',
    'general_tip': '🌟 Entertainment is all about personality and emotion. Show yourself!',
    'text_placement': 'Entertainment: Big, bold text that creates curiosity works best.'
  },
  'music': {
    'high_composition_tip': '🎵 Artistic, visually striking thumbnails = music clicks. Beautiful!',
    'low_composition_tip': '🎨 Music thumbnails can be more artistic and visually bold.',
    'high_color_tip': '✅ Music allows for creative color choices. Your palette is perfect!',
    'low_color_tip': '🌈 Music gives you freedom to be creative with colors. Try bolder choices!',
    'power_words_tip': '🎧 Words like "OFFICIAL", "MUSIC VIDEO", "LIVE" establish legitimacy.',
    'general_tip': '🎤 Music thumbnails blend artistry with branding. Great balance!',
    'text_placement': 'Music: Artist name + song title, keep it simple and readable.'
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
    'text_placement': 'Travel: Feature location name prominently, keep text minimal to showcase scenery.'
  },
  'general': {
    'balanced_tip': '📺 Your thumbnail shows balanced scoring across categories.',
    'improve_tip': '💡 Focus on your weakest scoring component for biggest improvement.',
    'power_words_tip': '📝 Strong power words increase clicks by up to 340%.',
    'general_tip': '✨ General content benefits from clear messaging and good composition.',
    'text_placement': 'General: Keep text simple, clear, and easy to read at small sizes.'
  }
};

/**
 * Get relevant insights based on niche and component scores
 */
export function getNicheInsights(nicheId: string, components: ComponentScores): string[] {
  const insights: string[] = [];
  const niche_tips = NICHE_INSIGHTS[nicheId] || NICHE_INSIGHTS['general'];
  
  // Add score-specific insights
  if (components.emotion && components.emotion > 80 && niche_tips.high_emotion_tip) {
    insights.push(niche_tips.high_emotion_tip);
  } else if (components.emotion && components.emotion < 40 && niche_tips.low_emotion_tip) {
    insights.push(niche_tips.low_emotion_tip);
  }
  
  if (components.color_pop && components.color_pop > 80 && niche_tips.high_color_tip) {
    insights.push(niche_tips.high_color_tip);
  } else if (components.color_pop && components.color_pop < 40 && niche_tips.low_color_tip) {
    insights.push(niche_tips.low_color_tip);
  }
  
  if (components.text_clarity && components.text_clarity > 80 && niche_tips.high_clarity_tip) {
    insights.push(niche_tips.high_clarity_tip);
  } else if (components.text_clarity && components.text_clarity < 40 && niche_tips.low_clarity_tip) {
    insights.push(niche_tips.low_clarity_tip);
  }
  
  if (components.composition && components.composition > 80 && niche_tips.high_composition_tip) {
    insights.push(niche_tips.high_composition_tip);
  } else if (components.composition && components.composition < 40 && niche_tips.low_composition_tip) {
    insights.push(niche_tips.low_composition_tip);
  }
  
  if (components.subject_prominence && components.subject_prominence > 80 && niche_tips.high_subject_tip) {
    insights.push(niche_tips.high_subject_tip);
  } else if (components.subject_prominence && components.subject_prominence < 40 && niche_tips.low_subject_tip) {
    insights.push(niche_tips.low_subject_tip);
  }
  
  // Always add power words and general tip
  if (niche_tips.power_words_tip) {
    insights.push(niche_tips.power_words_tip);
  }
  
  if (niche_tips.general_tip) {
    insights.push(niche_tips.general_tip);
  }
  
  // Add text placement advice
  if (niche_tips.text_placement) {
    insights.push(niche_tips.text_placement);
  }
  
  return insights.slice(0, 4); // Return top 4 most relevant insights
}