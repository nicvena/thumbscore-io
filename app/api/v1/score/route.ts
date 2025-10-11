import { NextRequest, NextResponse } from 'next/server';
import { ThumbnailRankingModel, MODEL_PRESETS } from '@/lib/ml-modeling';

// Initialize ML model (singleton for performance)
let mlModel: ThumbnailRankingModel | null = null;

async function getMLModel(): Promise<ThumbnailRankingModel> {
  if (!mlModel) {
    mlModel = new ThumbnailRankingModel(MODEL_PRESETS.production);
    console.log('[Inference Service] ML Model initialized with production preset');
  }
  return mlModel;
}

// Request/Response Types matching API contract
interface ScoreRequest {
  title: string;
  thumbnails: Array<{
    id: string;
    url: string;
  }>;
  category?: string;
}

interface ThumbnailScore {
  id: string;
  ctr_score: number;
  subscores: {
    clarity: number;
    subject_prominence: number;
    contrast_pop: number;
    emotion: number;
    hierarchy: number;
    title_match: number;
  };
  insights: string[];
  overlays: {
    saliency_heatmap_url: string;
    ocr_boxes_url: string;
    face_boxes_url: string;
  };
}

interface ScoreResponse {
  winner_id: string;
  thumbnails: ThumbnailScore[];
  explanation: string;
}

export async function POST(request: NextRequest) {
  const startTime = Date.now();
  
  try {
    const body: ScoreRequest = await request.json();
    const { title, thumbnails, category = 'general' } = body;
    
    // Check if Python service should be used
    const USE_PYTHON_SERVICE = process.env.USE_PYTHON_SERVICE === 'true';
    const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8000';
    
    if (USE_PYTHON_SERVICE) {
      console.log('[Inference] Proxying to Python FastAPI service...');
      
      try {
        const pythonResponse = await fetch(`${PYTHON_SERVICE_URL}/v1/score`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title, thumbnails, category })
        });
        
        if (!pythonResponse.ok) {
          throw new Error(`Python service error: ${pythonResponse.statusText}`);
        }
        
        const pythonData = await pythonResponse.json();
        const duration = Date.now() - startTime;
        
        console.log(`[Inference] Python service completed in ${duration}ms. Winner: ${pythonData.winner_id}`);
        
        return NextResponse.json(pythonData, {
          headers: {
            'X-Processing-Time-Ms': duration.toString(),
            'X-Model-Version': 'v3.0.0',
            'X-Model-Type': 'python-fastapi',
            'X-Inference-Backend': 'python'
          }
        });
      } catch (error) {
        console.error('[Inference] Python service failed, falling back to TypeScript:', error);
        // Fall through to TypeScript implementation
      }
    }

    // Validate request
    if (!title || !thumbnails || thumbnails.length === 0) {
      return NextResponse.json(
        { 
          error: 'Invalid request',
          message: 'title and thumbnails are required',
          example: {
            title: 'Video Title',
            thumbnails: [
              { id: 'A', url: 'https://...' },
              { id: 'B', url: 'https://...' }
            ],
            category: 'people-blogs'
          }
        },
        { status: 400 }
      );
    }

    if (thumbnails.length > 10) {
      return NextResponse.json(
        { error: 'Too many thumbnails', message: 'Maximum 10 thumbnails per request' },
        { status: 400 }
      );
    }

    console.log(`[Inference] Processing ${thumbnails.length} thumbnails for: "${title}"`);

    // Get ML model
    const model = await getMLModel();

    // Process each thumbnail
    const scoredThumbnails: ThumbnailScore[] = [];
    
    for (const thumbnail of thumbnails) {
      // 1. Download and encode thumbnail image
      const imageBuffer = await downloadImage(thumbnail.url);
      const embedding = await encodeImage(imageBuffer);
      
      // 2. Create metadata for model
      const metadata = {
        channelId: 'inference', // Unknown at inference time
        publishedAt: new Date().toISOString(),
        title,
        category,
        duration: 300
      };
      
      // 3. Get ML predictions
      const predictedCTR = await model.predictCTR(embedding, metadata);
      const subScores = await model.predictSubScores(embedding, metadata);
      
      // Convert to 0-100 scale and round
      const ctrScore = Math.round(Math.min(100, Math.max(0, predictedCTR * 100)));
      
      // Ensure subscores are in 0-100 range (model returns 0-1, convert to percentage)
      const subscores = {
        clarity: Math.round(Math.min(100, Math.max(0, subScores.clarity * 100))),
        subject_prominence: Math.round(Math.min(100, Math.max(0, subScores.subjectProminence * 100))),
        contrast_pop: Math.round(Math.min(100, Math.max(0, subScores.contrastColorPop * 100))),
        emotion: Math.round(Math.min(100, Math.max(0, subScores.emotion * 100))),
        hierarchy: Math.round(Math.min(100, Math.max(0, subScores.visualHierarchy * 100))),
        title_match: Math.round(Math.min(100, Math.max(0, subScores.clickIntentMatch * 100)))
      };
      
      // 4. Generate insights based on subscores
      const insights = generateInsights(subscores, title);
      
      // 5. Generate overlay URLs (visualization endpoints)
      const sessionId = generateSessionId();
      const overlays = {
        saliency_heatmap_url: `/api/v1/overlays/${sessionId}/${thumbnail.id}/heatmap.png`,
        ocr_boxes_url: `/api/v1/overlays/${sessionId}/${thumbnail.id}/ocr.png`,
        face_boxes_url: `/api/v1/overlays/${sessionId}/${thumbnail.id}/faces.png`
      };
      
      scoredThumbnails.push({
        id: thumbnail.id,
        ctr_score: ctrScore,
        subscores,
        insights,
        overlays
      });
      
      console.log(`[Inference] Thumbnail ${thumbnail.id}: CTR=${ctrScore}%, clarity=${subscores.clarity}`);
    }
    
    // Sort by CTR score (descending)
    scoredThumbnails.sort((a, b) => b.ctr_score - a.ctr_score);
    
    // Winner is the highest scoring thumbnail
    const winner = scoredThumbnails[0];
    const winnerId = winner.id;
    
    // Generate explanation
    const explanation = generateExplanation(winner, scoredThumbnails);
    
    const response: ScoreResponse = {
      winner_id: winnerId,
      thumbnails: scoredThumbnails,
      explanation
    };
    
    const duration = Date.now() - startTime;
    console.log(`[Inference] Completed in ${duration}ms. Winner: ${winnerId} (${winner.ctr_score}%)`);
    
    return NextResponse.json(response, {
      headers: {
        'X-Processing-Time-Ms': duration.toString(),
        'X-Model-Version': 'v3.0.0',
        'X-Model-Type': 'two-stage-multi-task'
      }
    });

  } catch (error) {
    console.error('[Inference] Error:', error);
    return NextResponse.json(
      { 
        error: 'Inference failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    service: 'Thumbnail Scoring Inference API',
    version: 'v1',
    status: 'operational',
    model: {
      type: 'Two-Stage Multi-Task Learning',
      version: '3.0.0',
      features: [
        'CLIP ViT-L/14 backbone',
        'Pairwise ranking head',
        'CTR prediction head',
        '6 auxiliary sub-score heads'
      ]
    },
    endpoints: {
      'POST /api/v1/score': 'Score and rank thumbnails',
      'GET /api/v1/score': 'API information',
      'GET /api/v1/overlays/:session/:id/:type': 'Get visualization overlays'
    },
    contract: {
      request: {
        title: 'string (required) - Video title',
        thumbnails: 'array (required) - List of {id, url}',
        category: 'string (optional) - Video category'
      },
      response: {
        winner_id: 'string - ID of highest scoring thumbnail',
        thumbnails: 'array - Scored thumbnails with insights',
        explanation: 'string - Why the winner was selected'
      }
    },
    limits: {
      max_thumbnails: 10,
      timeout_seconds: 30,
      rate_limit: '100 requests per minute'
    },
    example: {
      request: {
        title: "I Tried MrBeast's $1 vs $100,000 Plane Seat",
        thumbnails: [
          { id: 'A', url: 'https://example.com/thumb_a.jpg' },
          { id: 'B', url: 'https://example.com/thumb_b.jpg' },
          { id: 'C', url: 'https://example.com/thumb_c.jpg' }
        ],
        category: 'people-blogs'
      },
      response: {
        winner_id: 'B',
        thumbnails: [
          {
            id: 'B',
            ctr_score: 85,
            subscores: {
              clarity: 82,
              subject_prominence: 89,
              contrast_pop: 86,
              emotion: 88,
              hierarchy: 83,
              title_match: 84
            },
            insights: [
              'Excellent face prominence drives attention',
              'High contrast text ensures mobile readability',
              'Strong title-thumbnail semantic alignment'
            ],
            overlays: {
              saliency_heatmap_url: '/api/v1/overlays/session_id/B/heatmap.png',
              ocr_boxes_url: '/api/v1/overlays/session_id/B/ocr.png',
              face_boxes_url: '/api/v1/overlays/session_id/B/faces.png'
            }
          }
        ],
        explanation: 'B wins due to larger face prominence, high text contrast, and clear title alignment.'
      }
    }
  });
}

// Helper Functions

async function downloadImage(url: string): Promise<Buffer> {
  // Simulate image download
  // In production: fetch(url) and convert to buffer
  console.log(`[Inference] Downloading image: ${url}`);
  
  // Mock image buffer
  return Buffer.from(`mock-image-data-${url}`);
}

async function encodeImage(imageBuffer: Buffer): Promise<number[]> {
  // Simulate CLIP encoding
  // In production: run through CLIP model
  
  // Generate deterministic but varied embedding
  const hash = imageBuffer.toString().split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const embedding = Array.from({ length: 768 }, (_, i) => {
    const seed = (hash + i) % 1000000;
    return Math.sin(seed) * Math.cos(seed * 1.1);
  });
  
  return embedding;
}

function generateInsights(subscores: Record<string, number>, title: string): string[] {
  const insights: string[] = [];
  
  // Clarity insights
  if (subscores.clarity < 70) {
    const wordCount = title.split(' ').length;
    if (wordCount > 5) {
      insights.push(`Reduce words from ${wordCount}→3; use bold block font`);
    } else {
      insights.push('Increase text size and use high-contrast block font');
    }
    insights.push('Boost contrast between text and background');
  } else if (subscores.clarity >= 80) {
    insights.push('Excellent text clarity for mobile viewing');
  }
  
  // Subject prominence insights
  if (subscores.subject_prominence < 70) {
    insights.push('Increase subject size ~25%');
    insights.push('Center main subject for better attention capture');
  } else if (subscores.subject_prominence >= 85) {
    insights.push('Strong subject prominence drives attention');
  }
  
  // Contrast/Color insights
  if (subscores.contrast_pop < 70) {
    insights.push('Boost saturation by 15-20% for more visual pop');
    insights.push('Use complementary colors for better contrast');
  } else if (subscores.contrast_pop >= 80) {
    insights.push('High contrast ensures excellent visibility in feeds');
  }
  
  // Emotion insights
  if (subscores.emotion < 70) {
    insights.push('Add more expressive facial emotion or action');
    insights.push('Capture moment of peak emotion or surprise');
  } else if (subscores.emotion >= 85) {
    insights.push('Strong emotional appeal drives engagement');
  }
  
  // Hierarchy insights
  if (subscores.hierarchy < 70) {
    insights.push('Create clearer visual hierarchy with size contrast');
    insights.push('Reduce visual clutter - simplify composition');
  } else if (subscores.hierarchy >= 80) {
    insights.push('Well-balanced composition guides viewer attention');
  }
  
  // Title match insights
  if (subscores.title_match < 70) {
    insights.push('Better align thumbnail content with title theme');
    insights.push('Include visual element referenced in title');
  } else if (subscores.title_match >= 85) {
    insights.push('Strong title-thumbnail semantic alignment');
  }
  
  // If everything is good, acknowledge strengths
  if (insights.length === 0) {
    insights.push('Thumbnail performs well across all metrics');
    insights.push('Minor tweaks may yield marginal improvements');
  }
  
  // Limit to top 3-5 most actionable insights
  return insights.slice(0, Math.min(5, insights.length));
}

function generateExplanation(winner: ThumbnailScore, allThumbnails: ThumbnailScore[]): string {
  const reasons: string[] = [];
  
  // Find strongest subscores
  const subscoreNames = {
    clarity: 'text clarity',
    subject_prominence: 'face/subject prominence',
    contrast_pop: 'color contrast',
    emotion: 'emotional appeal',
    hierarchy: 'visual hierarchy',
    title_match: 'title alignment'
  };
  
  const topSubscores = Object.entries(winner.subscores)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3);
  
  for (const [key, score] of topSubscores) {
    if (score >= 80) {
      reasons.push(subscoreNames[key as keyof typeof subscoreNames]);
    }
  }
  
  // Compare to other thumbnails
  if (allThumbnails.length > 1) {
    const scoreDiff = winner.ctr_score - allThumbnails[1].ctr_score;
    if (scoreDiff > 10) {
      reasons.push('significantly higher overall CTR prediction');
    }
  }
  
  const reasonText = reasons.length > 0 
    ? reasons.slice(0, 3).join(', ')
    : 'better overall performance across metrics';
  
  return `${winner.id} wins due to ${reasonText}.`;
}

function generateSessionId(): string {
  return `inf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
