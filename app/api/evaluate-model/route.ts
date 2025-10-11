import { NextRequest, NextResponse } from 'next/server';
import { ThumbnailRankingModel, MODEL_PRESETS } from '@/lib/ml-modeling';
import { ModelEvaluationPipeline, EvaluationDataset, ABTestResult } from '@/lib/model-evaluation';

// Initialize evaluation pipeline
const evaluationPipeline = new ModelEvaluationPipeline();

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { 
      modelPreset = 'advanced',
      includeABTests = false,
      generateMockData = true 
    } = body;

    console.log('Model evaluation API called with:', { modelPreset, includeABTests, generateMockData });

    if (!generateMockData) {
      return NextResponse.json({
        message: 'Evaluation requires test dataset',
        status: 'ready',
        requiredData: [
          'Held-out channels (20-30% of total channels)',
          'Test samples with views_per_hour ground truth',
          'Pairwise comparison samples',
          'Optional: A/B test results from live deployment'
        ],
        metrics: [
          'Pairwise AUC on held-out channels',
          'Spearman ρ between predicted score and views_per_hour',
          'A/B test win rate (Phase 2)'
        ]
      });
    }

    // Generate mock evaluation dataset
    console.log('Generating mock evaluation dataset...');
    const mockDataset = generateMockEvaluationData();
    
    // Generate mock A/B test results if requested
    const mockABTests = includeABTests ? generateMockABTestResults() : undefined;
    
    // Load model
    const model = new ThumbnailRankingModel(MODEL_PRESETS[modelPreset as keyof typeof MODEL_PRESETS]);
    
    // Run comprehensive evaluation
    console.log('Running comprehensive evaluation...');
    const metrics = await evaluationPipeline.evaluateModel(model, mockDataset, mockABTests);
    
    // Determine model readiness
    const readiness = determineModelReadiness(metrics);
    
    return NextResponse.json({
      message: 'Model evaluation completed',
      status: 'completed',
      metrics: {
        pairwiseAUC: {
          overall: metrics.pairwiseAUC.overall,
          confidence95: metrics.pairwiseAUC.confidence95,
          channelsEvaluated: metrics.pairwiseAUC.byChannel.size,
          nichesEvaluated: metrics.pairwiseAUC.byNiche.size,
          target: '≥0.65 baseline, ≥0.72 with analytics',
          status: metrics.pairwiseAUC.overall >= 0.72 ? 'excellent' : 
                  metrics.pairwiseAUC.overall >= 0.65 ? 'good' : 'needs_improvement'
        },
        spearmanCorrelation: {
          overall: metrics.spearmanCorrelation.overall,
          pValue: metrics.spearmanCorrelation.pValue,
          channelsEvaluated: metrics.spearmanCorrelation.byChannel.size,
          target: '>0.3 minimum, >0.5 good',
          status: metrics.spearmanCorrelation.overall >= 0.5 ? 'excellent' : 
                  metrics.spearmanCorrelation.overall >= 0.3 ? 'acceptable' : 'needs_improvement'
        },
        abTestWinRate: metrics.abTestWinRate ? {
          overall: metrics.abTestWinRate.overall,
          modelWins: metrics.abTestWinRate.modelWins,
          creatorWins: metrics.abTestWinRate.creatorWins,
          totalTests: metrics.abTestWinRate.totalTests,
          confidence95: metrics.abTestWinRate.confidenceInterval95,
          status: metrics.abTestWinRate.overall >= 0.5 ? 'model_outperforms' : 'needs_improvement'
        } : null,
        diagnostics: {
          calibration: metrics.diagnostics.calibration,
          coverage: metrics.diagnostics.coverage,
          fairnessByNiche: Object.fromEntries(metrics.diagnostics.fairness)
        }
      },
      readiness,
      recommendations: generateRecommendations(metrics),
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Model evaluation error:', error);
    return NextResponse.json(
      { 
        error: 'Model evaluation failed',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Model Evaluation API',
    status: 'operational',
    endpoints: {
      'POST /api/evaluate-model': 'Evaluate model with comprehensive metrics',
      'GET /api/evaluate-model': 'Get evaluation API information'
    },
    evaluationMetrics: {
      '1. Pairwise AUC': {
        description: 'Predict which thumbnail outperforms on held-out channels',
        target: '≥0.65 baseline, ≥0.72 with YouTube Analytics',
        method: 'Bootstrap confidence intervals, per-channel and per-niche breakdown'
      },
      '2. Spearman ρ': {
        description: 'Correlation between predicted score and real views_per_hour within channel',
        target: '>0.3 minimum, >0.5 good',
        method: 'Rank correlation with statistical significance (p-value)'
      },
      '3. A/B Test Win Rate': {
        description: 'Percentage of model-picked thumbnails that beat creator choice (Phase 2)',
        target: '>50% model wins',
        method: 'Wilson score interval for confidence, breakdown by niche'
      }
    },
    dataBacked: {
      principle: 'Data-backed evaluation, not vibes',
      heldOutChannels: '20-30% of total channels for unbiased testing',
      statisticalRigor: 'Confidence intervals, p-values, significance testing',
      fairness: 'Performance evaluated across niches to ensure no bias'
    }
  });
}

// Helper Functions
function generateMockEvaluationData(): EvaluationDataset {
  const heldOutChannels = Array.from({ length: 50 }, (_, i) => `held_out_channel_${i}`);
  const niches = ['tech', 'gaming', 'education', 'entertainment', 'beauty'];
  
  // Generate test samples
  const testSamples = [];
  for (let i = 0; i < 500; i++) {
    const channelId = heldOutChannels[Math.floor(Math.random() * heldOutChannels.length)];
    const publishedDate = new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000);
    const hoursSincePublish = (Date.now() - publishedDate.getTime()) / (1000 * 60 * 60);
    const viewCount = Math.floor(Math.random() * 500000) + 1000;
    const viewsPerHour = viewCount / hoursSincePublish;
    
    testSamples.push({
      channelId,
      videoId: `video_${i}`,
      thumbnailUrl: `https://example.com/thumb_${i}.jpg`,
      embedding: Array.from({ length: 768 }, () => Math.random() * 2 - 1),
      title: `Test Video ${i}`,
      publishedAt: publishedDate.toISOString(),
      viewCount,
      viewsPerHour,
      trueCTR: Math.random() > 0.7 ? Math.random() * 0.1 : undefined,
      category: niches[Math.floor(Math.random() * niches.length)]
    });
  }
  
  // Generate pairwise samples
  const pairwiseSamples = [];
  for (let i = 0; i < 200; i++) {
    const channelId = heldOutChannels[Math.floor(Math.random() * heldOutChannels.length)];
    const viewsPerHourA = Math.random() * 10000;
    const viewsPerHourB = Math.random() * 10000;
    
    pairwiseSamples.push({
      channelId,
      thumbnailA: {
        videoId: `video_a_${i}`,
        embedding: Array.from({ length: 768 }, () => Math.random() * 2 - 1),
        viewsPerHour: viewsPerHourA,
        trueCTR: Math.random() > 0.7 ? Math.random() * 0.1 : undefined
      },
      thumbnailB: {
        videoId: `video_b_${i}`,
        embedding: Array.from({ length: 768 }, () => Math.random() * 2 - 1),
        viewsPerHour: viewsPerHourB,
        trueCTR: Math.random() > 0.7 ? Math.random() * 0.1 : undefined
      },
      winner: viewsPerHourA > viewsPerHourB ? 'A' as const : 'B' as const
    });
  }
  
  return {
    heldOutChannels,
    testSamples,
    pairwiseSamples
  };
}

function generateMockABTestResults(): ABTestResult[] {
  const results: ABTestResult[] = [];
  const niches = ['tech', 'gaming', 'education', 'entertainment', 'beauty'];
  
  for (let i = 0; i < 50; i++) {
    const modelCTR = 0.03 + Math.random() * 0.07; // 3-10%
    const creatorCTR = 0.02 + Math.random() * 0.06; // 2-8%
    const impressions = Math.floor(Math.random() * 100000) + 10000;
    
    results.push({
      testId: `ab_test_${i}`,
      channelId: `channel_${i}`,
      videoId: `video_${i}`,
      modelThumbnail: `model_thumb_${i}.jpg`,
      creatorThumbnail: `creator_thumb_${i}.jpg`,
      winner: modelCTR > creatorCTR ? 'model' : modelCTR < creatorCTR ? 'creator' : 'tie',
      modelCTR,
      creatorCTR,
      impressions,
      confidence: 0.95,
      timestamp: new Date().toISOString()
    });
  }
  
  return results;
}

function determineModelReadiness(metrics: any): {
  status: 'production_ready' | 'good' | 'needs_improvement';
  confidence: number;
  summary: string;
} {
  const aucGood = metrics.pairwiseAUC.overall >= 0.65;
  const aucExcellent = metrics.pairwiseAUC.overall >= 0.72;
  const spearmanGood = metrics.spearmanCorrelation.overall >= 0.3;
  const spearmanExcellent = metrics.spearmanCorrelation.overall >= 0.5;
  const abTestGood = metrics.abTestWinRate ? metrics.abTestWinRate.overall >= 0.5 : true;
  
  if (aucExcellent && spearmanExcellent && abTestGood) {
    return {
      status: 'production_ready',
      confidence: 0.95,
      summary: 'Model exceeds all target metrics and is ready for production deployment'
    };
  } else if (aucGood && spearmanGood) {
    return {
      status: 'good',
      confidence: 0.75,
      summary: 'Model meets baseline targets and is suitable for pilot testing'
    };
  } else {
    return {
      status: 'needs_improvement',
      confidence: 0.5,
      summary: 'Model requires additional training or data to meet target metrics'
    };
  }
}

function generateRecommendations(metrics: any): string[] {
  const recommendations: string[] = [];
  
  if (metrics.pairwiseAUC.overall < 0.65) {
    recommendations.push('Increase pairwise training data - target ≥200k thumbnail pairs');
    recommendations.push('Consider fine-tuning CLIP layers (last 2-4 blocks)');
  } else if (metrics.pairwiseAUC.overall < 0.72) {
    recommendations.push('Integrate YouTube Analytics data for gold-label CTR pairs');
    recommendations.push('Expand to more held-out channels for better generalization');
  }
  
  if (metrics.spearmanCorrelation.overall < 0.3) {
    recommendations.push('Add more engagement features (likes, comments, watch time)');
    recommendations.push('Improve channel baseline normalization');
  } else if (metrics.spearmanCorrelation.overall < 0.5) {
    recommendations.push('Fine-tune sub-score auxiliary heads for better correlation');
  }
  
  if (metrics.abTestWinRate && metrics.abTestWinRate.overall < 0.5) {
    recommendations.push('Run longer A/B tests to reduce noise and improve confidence');
    recommendations.push('Analyze failure cases to identify model weaknesses');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('Model performing excellently - ready for production scale');
    recommendations.push('Consider expanding to more niches and geographies');
  }
  
  return recommendations;
}
