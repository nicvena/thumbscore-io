"""
Niche-specific configuration for thumbnail scoring optimization.

Each niche has customized weights, power words, visual preferences, and baseline scores
to provide more accurate thumbnail analysis based on content type.
"""

NICHES = {
    'gaming': {
        'id': 'gaming',
        'name': 'Gaming',
        'emoji': '🎮',
        'description': 'Video games, streaming, esports, gaming content',
        'weights': {
            'emotion': 0.25,
            'color_pop': 0.20,
            'power_words': 0.20,
            'text_clarity': 0.15,
            'subject_prominence': 0.10,
            'similarity': 0.10
        },
        'power_words': {
            'tier1': ['insane', 'broken', 'op', 'destroyed', 'unbeatable', 'clutch', 'epic', 'legendary'],
            'tier2': ['pro', 'ultimate', 'best', 'crazy', 'impossible', 'secret', 'hidden'],
            'tier3': ['new', 'first', 'world', 'record', 'challenge', 'vs', 'reaction']
        },
        'visual_preferences': {
            'saturation': 'high',  # 70-100%
            'contrast': 'extreme',  # High contrast preferred
            'face_size': 'large',  # 30-50% of frame
            'text_style': 'bold_outlined'
        },
        'baseline_score': 78,
        'faiss_index': 'indices/gaming_index.faiss'
    },
    'business': {
        'id': 'business',
        'name': 'Business & Finance',
        'emoji': '💼',
        'description': 'Entrepreneurship, investing, career advice, business growth',
        'weights': {
            'text_clarity': 0.30,
            'composition': 0.25,
            'similarity': 0.25,
            'color_pop': 0.10,
            'subject_prominence': 0.05,
            'emotion': 0.05
        },
        'power_words': {
            'tier1': ['proven', 'strategy', 'framework', 'blueprint', 'system', 'growth'],
            'tier2': ['scale', 'profit', 'revenue', 'success', 'million', 'secrets'],
            'tier3': ['tips', 'guide', 'how to', 'steps', 'avoid', 'mistakes']
        },
        'visual_preferences': {
            'saturation': 'low',  # 30-60%
            'contrast': 'moderate',
            'face_size': 'medium',  # 15-25% of frame
            'text_style': 'clean_sans_serif'
        },
        'baseline_score': 75,
        'faiss_index': 'indices/business_index.faiss'
    },
    'education': {
        'id': 'education',
        'name': 'Education & How-To',
        'emoji': '🎓',
        'description': 'Tutorials, courses, educational content, how-to guides',
        'weights': {
            'text_clarity': 0.30,
            'similarity': 0.25,
            'composition': 0.20,
            'color_pop': 0.15,
            'subject_prominence': 0.05,
            'emotion': 0.05
        },
        'power_words': {
            'tier1': ['learn', 'master', 'complete', 'guide', 'course', 'explained'],
            'tier2': ['easy', 'simple', 'beginner', 'advanced', 'tutorial', 'lesson'],
            'tier3': ['how to', 'step by step', 'ultimate', 'full', 'free', 'quick']
        },
        'visual_preferences': {
            'saturation': 'moderate',
            'contrast': 'moderate',
            'face_size': 'medium',
            'text_style': 'clear_readable'
        },
        'baseline_score': 72,
        'faiss_index': 'indices/education_index.faiss'
    },
    'tech': {
        'id': 'tech',
        'name': 'Tech & Reviews',
        'emoji': '💻',
        'description': 'Technology reviews, gadgets, software, tech news',
        'weights': {
            'text_clarity': 0.25,
            'composition': 0.25,
            'similarity': 0.25,
            'color_pop': 0.15,
            'subject_prominence': 0.10,
            'emotion': 0.00
        },
        'power_words': {
            'tier1': ['review', 'vs', 'unboxing', 'worth it', 'best', 'ultimate'],
            'tier2': ['new', 'first', 'hands on', 'comparison', 'honest', 'real'],
            'tier3': ['tech', 'features', 'specs', 'performance', 'price', 'buy']
        },
        'visual_preferences': {
            'saturation': 'moderate',
            'contrast': 'high',
            'face_size': 'small',  # Often no face, product focus
            'text_style': 'modern_clean'
        },
        'baseline_score': 74,
        'faiss_index': 'indices/tech_index.faiss'
    },
    'food': {
        'id': 'food',
        'name': 'Food & Cooking',
        'emoji': '🍳',
        'description': 'Recipes, cooking tutorials, food reviews, baking',
        'weights': {
            'color_pop': 0.30,
            'composition': 0.25,
            'text_clarity': 0.20,
            'similarity': 0.20,
            'emotion': 0.05,
            'subject_prominence': 0.00
        },
        'power_words': {
            'tier1': ['easy', 'quick', 'perfect', 'secret', 'authentic', 'delicious'],
            'tier2': ['recipe', 'homemade', 'crispy', 'fluffy', 'moist', 'creamy'],
            'tier3': ['minutes', 'simple', 'best', 'ultimate', 'restaurant', 'amazing']
        },
        'visual_preferences': {
            'saturation': 'high',
            'contrast': 'moderate',
            'face_size': 'none',  # Food close-up preferred
            'text_style': 'appetizing'
        },
        'baseline_score': 76,
        'faiss_index': 'indices/food_index.faiss'
    },
    'fitness': {
        'id': 'fitness',
        'name': 'Fitness & Health',
        'emoji': '💪',
        'description': 'Workouts, fitness tips, nutrition, weight loss, transformation',
        'weights': {
            'subject_prominence': 0.25,
            'composition': 0.20,
            'text_clarity': 0.20,
            'emotion': 0.15,
            'similarity': 0.15,
            'color_pop': 0.05
        },
        'power_words': {
            'tier1': ['transformation', 'lose', 'gain', 'workout', 'fast', 'days'],
            'tier2': ['results', 'proven', 'simple', 'easy', 'home', 'no equipment'],
            'tier3': ['body', 'muscle', 'fat', 'diet', 'weight', 'fitness']
        },
        'visual_preferences': {
            'saturation': 'moderate',
            'contrast': 'high',
            'face_size': 'medium',
            'text_style': 'bold_motivational'
        },
        'baseline_score': 73,
        'faiss_index': 'indices/fitness_index.faiss'
    },
    'entertainment': {
        'id': 'entertainment',
        'name': 'Entertainment & Vlogs',
        'emoji': '🎬',
        'description': 'Vlogs, entertainment, comedy, lifestyle content',
        'weights': {
            'emotion': 0.30,
            'color_pop': 0.20,
            'similarity': 0.20,
            'text_clarity': 0.15,
            'subject_prominence': 0.10,
            'composition': 0.05
        },
        'power_words': {
            'tier1': ['shocking', 'revealed', 'secret', 'exposed', 'truth', 'crazy'],
            'tier2': ['story', 'reaction', 'first time', 'never', 'finally', 'real'],
            'tier3': ['life', 'day', 'vlog', 'behind', 'what', 'why']
        },
        'visual_preferences': {
            'saturation': 'high',
            'contrast': 'high',
            'face_size': 'large',
            'text_style': 'bold_expressive'
        },
        'baseline_score': 77,
        'faiss_index': 'indices/entertainment_index.faiss'
    },
    'music': {
        'id': 'music',
        'name': 'Music',
        'emoji': '🎵',
        'description': 'Music videos, covers, tutorials, music production',
        'weights': {
            'composition': 0.30,
            'color_pop': 0.25,
            'similarity': 0.20,
            'text_clarity': 0.15,
            'emotion': 0.05,
            'subject_prominence': 0.05
        },
        'power_words': {
            'tier1': ['official', 'music video', 'cover', 'remix', 'ft', 'live'],
            'tier2': ['new', 'album', 'single', 'premiere', 'exclusive', 'first'],
            'tier3': ['lyrics', 'audio', 'visualizer', 'behind', 'making', 'studio']
        },
        'visual_preferences': {
            'saturation': 'high',
            'contrast': 'artistic',
            'face_size': 'medium',
            'text_style': 'artistic_bold'
        },
        'baseline_score': 79,
        'faiss_index': 'indices/music_index.faiss'
    },
    'travel': {
        'id': 'travel',
        'name': 'Travel & Lifestyle',
        'emoji': '✈️',
        'description': 'Travel vlogs, destination guides, adventure content',
        'weights': {
            'composition': 0.30,      # Stunning visuals are KEY
            'color_pop': 0.25,        # Vibrant landscapes
            'text_clarity': 0.20,     # Location names need to be clear
            'similarity': 0.15,
            'emotion': 0.05,          # Face less important than scenery
            'subject_prominence': 0.05
        },
        'power_words': {
            'tier1': ['hidden', 'secret', 'paradise', 'ultimate', 'guide', 'best'],
            'tier2': ['travel', 'adventure', 'explore', 'journey', 'destination', 'tips'],
            'tier3': ['budget', 'cheap', 'expensive', 'worth it', 'solo', 'backpacking']
        },
        'visual_preferences': {
            'saturation': 'high',     # Vibrant, aspirational
            'contrast': 'moderate',
            'face_size': 'small',     # Scenery is the star
            'text_style': 'clean_bold'
        },
        'baseline_score': 76,
        'faiss_index': 'indices/travel_index.faiss'
    },
    'general': {
        'id': 'general',
        'name': 'General / Other',
        'emoji': '📺',
        'description': 'General content or mixed categories',
        'weights': {
            'similarity': 0.30,
            'text_clarity': 0.20,
            'composition': 0.20,
            'color_pop': 0.15,
            'emotion': 0.10,
            'subject_prominence': 0.05
        },
        'power_words': {
            'tier1': ['best', 'top', 'ultimate', 'complete', 'secrets', 'revealed'],
            'tier2': ['new', 'how to', 'guide', 'tips', 'tricks', 'hacks'],
            'tier3': ['watch', 'must', 'you need', 'everyone', 'everything', 'full']
        },
        'visual_preferences': {
            'saturation': 'moderate',
            'contrast': 'moderate',
            'face_size': 'medium',
            'text_style': 'balanced'
        },
        'baseline_score': 70,
        'faiss_index': 'indices/general_index.faiss'
    }
}

# Helper function to get niche config
def get_niche_config(niche_id):
    """Get configuration for a specific niche, fallback to general if not found."""
    return NICHES.get(niche_id, NICHES['general'])

# Get list of all niches for dropdown
def get_all_niches():
    """Return list of all available niches for frontend dropdown."""
    return [
        {
            'id': n['id'], 
            'name': n['name'], 
            'emoji': n['emoji'], 
            'description': n['description']
        }
        for n in NICHES.values()
    ]

# Validate niche ID
def is_valid_niche(niche_id):
    """Check if niche_id is valid."""
    return niche_id in NICHES

# Get niche-specific baseline score
def get_niche_baseline(niche_id):
    """Get baseline score for niche calibration."""
    config = get_niche_config(niche_id)
    return config['baseline_score']