import { NextRequest, NextResponse } from 'next/server';
import { ThumbnailModelTraining } from '@/lib/model-training';

// Initialize training pipeline
const modelTraining = new ThumbnailModelTraining();

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      modelPreset = 'advanced',
      enableTraining = false 
    } = body;

    console.log('Model training API called with:', { modelPreset, enableTraining });

    if (!enableTraining) {
      return NextResponse.json({
        message: 'Training simulation mode',
        status: 'ready',
        modelPreset,
        features: [
          'Two-stage multi-task learning',
          'Pairwise Ranking Head with Margin Ranking Loss',
          'Absolute CTR Head with channel normalization',
          'Auxiliary Heads for sub-scores (clarity, contrast, emotion, etc.)',
          'CLIP ViT-L/14 backbone (freeze/fine-tune options)',
          'Label smoothing + mixup augmentation',
          'Stochastic Weight Averaging for stable inference'
        ],
        targetMetrics: {
          pairwiseAUC: '≥0.65 baseline, ≥0.72 with analytics data',
          ctrCorrelation: '>0.3 minimum, >0.5 good',
          trainingData: '120k-200k thumbnails across 500-1,500 channels'
        },
        architecture: {
          backbone: 'CLIP ViT-L/14 (frozen v1, fine-tune last 2-4 blocks v2)',
          rankingHead: 'Margin Ranking Loss for pairwise comparison',
          ctrHead: 'Normalized CTR prediction with channel baseline',
          auxiliaryHeads: '6 sub-scores for explainable recommendations',
          training: 'Label smoothing (0.1-0.2) + mixup (α=0.2-0.4) + SWA'
        }
      });
    }

    // Start actual training (simulation)
    console.log('Starting model training simulation...');
    
    const trainingResults = await modelTraining.runFullTrainingPipeline();
    
    return NextResponse.json({
      message: 'Model training completed successfully',
      status: 'completed',
      results: {
        modelPath: trainingResults.modelPath,
        metrics: trainingResults.metrics,
        timestamp: trainingResults.timestamp,
        config: trainingResults.config
      },
      summary: {
        pairwiseAUC: trainingResults.metrics.pairwiseAUC,
        ctrPerformance: {
          mse: trainingResults.metrics.ctrMSE,
          correlation: trainingResults.metrics.ctrCorrelation
        },
        readiness: trainingResults.metrics.pairwiseAUC > 0.65 ? 'Production Ready' : 'Needs Improvement'
      }
    });

  } catch (error) {
    console.error('Model training error:', error);
    return NextResponse.json(
      { 
        error: 'Model training failed',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Model Training API',
    status: 'operational',
    endpoints: {
      'POST /api/train-model': 'Train or simulate model training',
      'GET /api/train-model': 'Get training API information'
    },
    modelArchitecture: {
      type: 'Two-stage Multi-task Learning',
      components: [
        'Pairwise Ranking Head - Margin Ranking Loss',
        'Absolute CTR Head - Normalized prediction',
        'Auxiliary Heads - 6 explainable sub-scores',
        'CLIP ViT-L/14 backbone with freeze/fine-tune options'
      ],
      trainingFeatures: [
        'Label smoothing (0.1-0.2)',
        'Mixup augmentation (α=0.2-0.4)',
        'Stochastic Weight Averaging',
        'Channel baseline normalization',
        'Pairwise training within ±30 days'
      ]
    },
    targetMetrics: {
      pairwiseAUC: {
        baseline: '≥0.65',
        target: '≥0.72 with YouTube Analytics data',
        description: 'Predict which thumbnail outperforms in same channel/time window'
      },
      ctrCorrelation: {
        minimum: '>0.3',
        good: '>0.5',
        description: 'Correlation between predicted and actual CTR'
      },
      dataRequirements: {
        samples: '120k-200k thumbnails',
        channels: '500-1,500 channels',
        niches: 'tech, beauty, gaming, education, vlog, news',
        timeWindow: '±30 days for pairwise comparison'
      }
    }
  });
}
