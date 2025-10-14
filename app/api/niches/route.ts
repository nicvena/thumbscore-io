import { NextResponse } from 'next/server';

// Niche configuration - mirrors Python backend config
const NICHES = {
  gaming: {
    id: 'gaming',
    name: 'Gaming',
    emoji: '🎮',
    description: 'Video games, streaming, esports, gaming content'
  },
  business: {
    id: 'business',
    name: 'Business & Finance',
    emoji: '💼',
    description: 'Entrepreneurship, investing, career advice, business growth'
  },
  education: {
    id: 'education',
    name: 'Education & How-To',
    emoji: '🎓',
    description: 'Tutorials, courses, educational content, how-to guides'
  },
  tech: {
    id: 'tech',
    name: 'Tech & Reviews',
    emoji: '💻',
    description: 'Technology reviews, gadgets, software, tech news'
  },
  food: {
    id: 'food',
    name: 'Food & Cooking',
    emoji: '🍳',
    description: 'Recipes, cooking tutorials, food reviews, baking'
  },
  fitness: {
    id: 'fitness',
    name: 'Fitness & Health',
    emoji: '💪',
    description: 'Workouts, fitness tips, nutrition, weight loss, transformation'
  },
  entertainment: {
    id: 'entertainment',
    name: 'Entertainment & Vlogs',
    emoji: '🎬',
    description: 'Vlogs, entertainment, comedy, lifestyle content'
  },
  music: {
    id: 'music',
    name: 'Music',
    emoji: '🎵',
    description: 'Music videos, covers, tutorials, music production'
  },
  general: {
    id: 'general',
    name: 'General / Other',
    emoji: '📺',
    description: 'General content or mixed categories'
  }
};

export async function GET() {
  try {
    // Get all available niches for dropdown
    const niches = Object.values(NICHES);

    return NextResponse.json({
      success: true,
      niches: niches,
      count: niches.length
    });

  } catch (error) {
    console.error('Error fetching niches:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to fetch niches',
        niches: [] 
      },
      { status: 500 }
    );
  }
}