import { NextRequest, NextResponse } from 'next/server';

// Visualization overlay endpoint
export async function GET(
  request: NextRequest,
  { params }: { params: { session: string; id: string; type: string } }
) {
  const { session, id, type } = params;
  
  console.log(`[Overlays] Generating ${type} visualization for ${id} (session: ${session})`);
  
  // Validate type
  const validTypes = ['heatmap.png', 'ocr.png', 'faces.png'];
  if (!validTypes.includes(type)) {
    return NextResponse.json(
      { error: 'Invalid overlay type', valid: validTypes },
      { status: 400 }
    );
  }
  
  // In production: Generate actual visualization
  // For now, return placeholder or redirect to stored visualization
  
  const overlayType = type.replace('.png', '');
  
  return NextResponse.json({
    session,
    thumbnail_id: id,
    type: overlayType,
    status: 'available',
    message: 'Visualization overlay ready',
    visualization: {
      heatmap: {
        description: 'Saliency map showing attention hotspots',
        features: ['Rule of thirds alignment', 'Attention gradient', 'Focal point intensity']
      },
      ocr: {
        description: 'Text detection with bounding boxes',
        features: ['Text content', 'Confidence scores', 'Contrast analysis', 'Readability metrics']
      },
      faces: {
        description: 'Face detection with emotion analysis',
        features: ['Face bounding boxes', 'Emotion scores', 'Gaze direction', 'Expression intensity']
      }
    }[overlayType],
    download_url: `/api/v1/overlays/download/${session}/${id}/${type}`,
    note: 'In production: This would return actual PNG image with visual overlays'
  });
}

// Download endpoint for actual overlay images
export async function POST(
  request: NextRequest,
  { params }: { params: { session: string; id: string; type: string } }
) {
  // Generate and cache overlay visualization
  const { session, id, type } = params;
  
  // In production: Use canvas or image processing library to draw overlays
  // 1. Load original thumbnail
  // 2. Draw visualization elements (heatmap, boxes, etc.)
  // 3. Return as PNG
  
  return NextResponse.json({
    message: 'Overlay generation endpoint',
    session,
    thumbnail_id: id,
    type,
    note: 'POST to trigger visualization generation with custom parameters'
  });
}
