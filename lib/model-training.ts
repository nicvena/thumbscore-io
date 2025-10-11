/**
 * Model Training Infrastructure for YouTube Thumbnail Analysis
 * 
 * Handles data preparation, training orchestration, and model deployment
 */

import { ThumbnailRankingModel, ModelTrainingPipeline, MODEL_PRESETS, PairwiseInput, PairwiseTarget, CTRTarget, ThumbnailMetadata } from './ml-modeling';

// Training Data Types
export interface TrainingDataset {
  pairwiseData: Array<{
    input: PairwiseInput;
    target: PairwiseTarget;
  }>;
  ctrData: Array<{
    embedding: number[];
    metadata: ThumbnailMetadata;
    target: CTRTarget;
  }>;
  metadata: {
    totalSamples: number;
    channels: number;
    dateRange: { start: string; end: string };
    niches: string[];
  };
}

export interface TrainingConfig {
  modelPreset: keyof typeof MODEL_PRESETS;
  dataPath: string;
  outputPath: string;
  validationSplit: number;
  testSplit: number;
  enableEarlyStopping: boolean;
  earlyStoppingPatience: number;
  enableCheckpointing: boolean;
  checkpointInterval: number;
}

export interface TrainingResults {
  modelPath: string;
  metrics: {
    pairwiseAUC: number;
    ctrMSE: number;
    ctrMAE: number;
    ctrCorrelation: number;
  };
  trainingHistory: any[];
  config: TrainingConfig;
  timestamp: string;
}

// Data Preparation
export class TrainingDataPreparator {
  
  async preparePairwiseData(
    rawData: Array<{
      channelId: string;
      videoId: string;
      thumbnailUrl: string;
      title: string;
      publishedAt: string;
      viewCount: number;
      likeCount: number;
      commentCount: number;
      category: string;
      duration: number;
      trueCTR?: number;
    }>,
    timeWindowDays: number = 30
  ): Promise<Array<{input: PairwiseInput, target: PairwiseTarget}>> {
    
    console.log(`Preparing pairwise data from ${rawData.length} videos...`);
    
    const pairwiseData = [];
    
    // Group by channel
    const channelGroups = this.groupByChannel(rawData);
    
    for (const [channelId, videos] of channelGroups.entries()) {
      if (videos.length < 2) continue;
      
      // Sort by publish date
      videos.sort((a, b) => new Date(a.publishedAt).getTime() - new Date(b.publishedAt).getTime());
      
      // Create pairs within time window
      for (let i = 0; i < videos.length; i++) {
        for (let j = i + 1; j < videos.length; j++) {
          const videoA = videos[i];
          const videoB = videos[j];
          
          // Check time window constraint
          const daysDiff = (new Date(videoB.publishedAt).getTime() - new Date(videoA.publishedAt).getTime()) / (1000 * 60 * 60 * 24);
          if (daysDiff > timeWindowDays) break;
          
          // Generate embeddings (simulated)
          const embeddingA = await this.generateEmbedding(videoA.thumbnailUrl, videoA.title);
          const embeddingB = await this.generateEmbedding(videoB.thumbnailUrl, videoB.title);
          
          // Determine winner based on engagement proxy
          const proxyScoreA = this.computeEngagementProxy(videoA);
          const proxyScoreB = this.computeEngagementProxy(videoB);
          
          const winner = proxyScoreA > proxyScoreB ? 'A' : 'B';
          const margin = Math.abs(proxyScoreA - proxyScoreB);
          const proxyScore = Math.max(proxyScoreA, proxyScoreB);
          
          const input: PairwiseInput = {
            thumbnailA: {
              embedding: embeddingA,
              metadata: {
                channelId: videoA.channelId,
                publishedAt: videoA.publishedAt,
                title: videoA.title,
                category: videoA.category,
                duration: videoA.duration,
                viewCount: videoA.viewCount,
                likeCount: videoA.likeCount,
                commentCount: videoA.commentCount,
                trueCTR: videoA.trueCTR
              }
            },
            thumbnailB: {
              embedding: embeddingB,
              metadata: {
                channelId: videoB.channelId,
                publishedAt: videoB.publishedAt,
                title: videoB.title,
                category: videoB.category,
                duration: videoB.duration,
                viewCount: videoB.viewCount,
                likeCount: videoB.likeCount,
                commentCount: videoB.commentCount,
                trueCTR: videoB.trueCTR
              }
            },
            channelId,
            timeWindow: Math.round(daysDiff)
          };
          
          const target: PairwiseTarget = {
            winner,
            margin,
            proxyScore
          };
          
          pairwiseData.push({ input, target });
        }
      }
    }
    
    console.log(`Generated ${pairwiseData.length} pairwise samples`);
    return pairwiseData;
  }
  
  async prepareCTRData(
    rawData: Array<{
      channelId: string;
      videoId: string;
      thumbnailUrl: string;
      title: string;
      publishedAt: string;
      viewCount: number;
      likeCount: number;
      commentCount: number;
      category: string;
      duration: number;
      trueCTR?: number;
    }>
  ): Promise<Array<{embedding: number[], metadata: ThumbnailMetadata, target: CTRTarget}>> {
    
    console.log(`Preparing CTR data from ${rawData.length} videos...`);
    
    // Compute channel baselines
    const channelBaselines = this.computeChannelBaselines(rawData);
    
    const ctrData = [];
    
    for (const video of rawData) {
      const embedding = await this.generateEmbedding(video.thumbnailUrl, video.title);
      
      const metadata: ThumbnailMetadata = {
        channelId: video.channelId,
        publishedAt: video.publishedAt,
        title: video.title,
        category: video.category,
        duration: video.duration,
        viewCount: video.viewCount,
        likeCount: video.likeCount,
        commentCount: video.commentCount,
        trueCTR: video.trueCTR
      };
      
      // Normalize CTR
      const channelBaseline = channelBaselines.get(video.channelId)!;
      const proxyScore = this.computeEngagementProxy(video);
      const normalizedCTR = (proxyScore - channelBaseline.mean) / channelBaseline.std;
      
      const target: CTRTarget = {
        normalizedCTR: Math.max(0, Math.min(1, (normalizedCTR + 3) / 6)), // Scale to 0-1
        channelBaseline: normalizedCTR,
        trueCTR: video.trueCTR
      };
      
      ctrData.push({ embedding, metadata, target });
    }
    
    console.log(`Prepared ${ctrData.length} CTR samples`);
    return ctrData;
  }
  
  private groupByChannel(data: any[]): Map<string, any[]> {
    const groups = new Map<string, any[]>();
    
    for (const item of data) {
      if (!groups.has(item.channelId)) {
        groups.set(item.channelId, []);
      }
      groups.get(item.channelId)!.push(item);
    }
    
    return groups;
  }
  
  private computeChannelBaselines(data: any[]): Map<string, {mean: number, std: number}> {
    const channelGroups = this.groupByChannel(data);
    const baselines = new Map<string, {mean: number, std: number}>();
    
    for (const [channelId, videos] of channelGroups.entries()) {
      const scores = videos.map(video => this.computeEngagementProxy(video));
      const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
      const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
      const std = Math.sqrt(variance);
      
      baselines.set(channelId, { mean, std });
    }
    
    return baselines;
  }
  
  private computeEngagementProxy(video: any): number {
    // views_per_hour proxy
    const hoursSincePublish = (Date.now() - new Date(video.publishedAt).getTime()) / (1000 * 60 * 60);
    const viewsPerHour = video.viewCount / Math.max(1, hoursSincePublish);
    
    // Normalize by channel size and video duration
    const durationBonus = Math.min(1, video.duration / 600); // 10 minutes = 1.0
    const likeRatio = video.likeCount / Math.max(1, video.viewCount);
    const commentRatio = video.commentCount / Math.max(1, video.viewCount);
    
    return viewsPerHour * (1 + durationBonus * 0.1) * (1 + likeRatio * 0.2) * (1 + commentRatio * 0.1);
  }
  
  private async generateEmbedding(thumbnailUrl: string, title: string): Promise<number[]> {
    // Simulate CLIP embedding generation
    // In production: fetch image, run through CLIP model
    const imageHash = this.simpleHash(thumbnailUrl);
    const titleHash = this.simpleHash(title);
    
    // Generate deterministic but varied embeddings
    const embedding = Array.from({ length: 768 }, (_, i) => {
      const seed = (imageHash + titleHash + i) % 1000000;
      return (Math.sin(seed) * Math.cos(seed * 1.1) + Math.random() * 0.1);
    });
    
    return embedding;
  }
  
  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }
}

// Training Orchestrator
export class ModelTrainer {
  
  async trainModel(config: TrainingConfig): Promise<TrainingResults> {
    console.log(`Starting model training with preset: ${config.modelPreset}`);
    
    // Load training data
    const rawData = await this.loadTrainingData(config.dataPath);
    
    // Prepare data
    const preparator = new TrainingDataPreparator();
    const [pairwiseData, ctrData] = await Promise.all([
      preparator.preparePairwiseData(rawData),
      preparator.prepareCTRData(rawData)
    ]);
    
    // Create training pipeline
    const modelConfig = MODEL_PRESETS[config.modelPreset];
    const pipeline = new ModelTrainingPipeline(modelConfig);
    
    // Train model
    const results = await pipeline.train(pairwiseData, ctrData, config.validationSplit);
    
    // Save model
    const modelPath = await this.saveModel(pipeline, config.outputPath);
    
    // Prepare results
    const trainingResults: TrainingResults = {
      modelPath,
      metrics: {
        pairwiseAUC: results.pairwiseAUC,
        ctrMSE: results.ctrMetrics.mse,
        ctrMAE: results.ctrMetrics.mae,
        ctrCorrelation: results.ctrMetrics.correlation
      },
      trainingHistory: results.trainingHistory,
      config,
      timestamp: new Date().toISOString()
    };
    
    // Save training results
    await this.saveTrainingResults(trainingResults, config.outputPath);
    
    console.log(`Training completed. Pairwise AUC: ${results.pairwiseAUC.toFixed(4)}, CTR MSE: ${results.ctrMetrics.mse.toFixed(4)}`);
    
    return trainingResults;
  }
  
  private async loadTrainingData(dataPath: string): Promise<any[]> {
    // In production: load from database or file
    console.log(`Loading training data from ${dataPath}...`);
    
    // Simulate loading 10k+ samples
    const sampleData = [];
    const channels = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5'];
    const categories = ['tech', 'gaming', 'education', 'entertainment', 'news'];
    
    for (let i = 0; i < 10000; i++) {
      const channelId = channels[Math.floor(Math.random() * channels.length)];
      const publishedDate = new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000);
      
      sampleData.push({
        channelId,
        videoId: `video_${i}`,
        thumbnailUrl: `https://example.com/thumb_${i}.jpg`,
        title: `Sample Video Title ${i}`,
        publishedAt: publishedDate.toISOString(),
        viewCount: Math.floor(Math.random() * 1000000) + 1000,
        likeCount: Math.floor(Math.random() * 100000) + 100,
        commentCount: Math.floor(Math.random() * 10000) + 10,
        category: categories[Math.floor(Math.random() * categories.length)],
        duration: Math.floor(Math.random() * 1800) + 60, // 1-30 minutes
        trueCTR: Math.random() > 0.7 ? Math.random() * 0.1 : undefined // 30% have true CTR
      });
    }
    
    return sampleData;
  }
  
  private async saveModel(pipeline: ModelTrainingPipeline, outputPath: string): Promise<string> {
    const modelPath = `${outputPath}/model_${Date.now()}`;
    console.log(`Saving model to ${modelPath}...`);
    
    // In production: save actual model weights
    // await pipeline.save(modelPath);
    
    return modelPath;
  }
  
  private async saveTrainingResults(results: TrainingResults, outputPath: string): Promise<void> {
    const resultsPath = `${outputPath}/training_results_${Date.now()}.json`;
    console.log(`Saving training results to ${resultsPath}...`);
    
    // In production: save to file system or database
    // await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));
  }
}

// Model Evaluation and Deployment
export class ModelEvaluator {
  
  async evaluateModel(modelPath: string, testData: TrainingDataset): Promise<{
    pairwiseAUC: number;
    ctrMetrics: {mse: number, mae: number, correlation: number};
    subScoreMetrics: Record<string, {mse: number, mae: number}>;
    recommendations: string[];
  }> {
    console.log(`Evaluating model at ${modelPath}...`);
    
    // Load model
    const model = await this.loadModel(modelPath);
    
    // Evaluate pairwise ranking
    const pairwiseAUC = await model.evaluatePairwiseAUC(testData.pairwiseData.map(d => d));
    
    // Evaluate CTR prediction
    const ctrMetrics = await model.evaluateCTRMetrics(testData.ctrData.map(d => d));
    
    // Evaluate sub-scores
    const subScoreMetrics = await this.evaluateSubScores(model, testData.ctrData);
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(pairwiseAUC, ctrMetrics);
    
    return {
      pairwiseAUC,
      ctrMetrics,
      subScoreMetrics,
      recommendations
    };
  }
  
  private async loadModel(modelPath: string): Promise<ThumbnailRankingModel> {
    // In production: load actual model
    const config = MODEL_PRESETS.production;
    return new ThumbnailRankingModel(config);
  }
  
  private async evaluateSubScores(model: ThumbnailRankingModel, testData: any[]): Promise<Record<string, {mse: number, mae: number}>> {
    const subScoreMetrics: Record<string, {mse: number, mae: number}> = {};
    
    // Simulate sub-score evaluation
    const subScoreNames = ['clarity', 'subjectProminence', 'contrastColorPop', 'emotion', 'visualHierarchy', 'clickIntentMatch'];
    
    for (const scoreName of subScoreNames) {
      // Generate mock predictions and targets
      const predictions = testData.map(() => Math.random());
      const targets = testData.map(() => Math.random());
      
      const mse = predictions.reduce((sum, pred, i) => sum + Math.pow(pred - targets[i], 2), 0) / predictions.length;
      const mae = predictions.reduce((sum, pred, i) => sum + Math.abs(pred - targets[i]), 0) / predictions.length;
      
      subScoreMetrics[scoreName] = { mse, mae };
    }
    
    return subScoreMetrics;
  }
  
  private generateRecommendations(pairwiseAUC: number, ctrMetrics: any): string[] {
    const recommendations = [];
    
    if (pairwiseAUC < 0.65) {
      recommendations.push('Pairwise AUC below target (0.65). Consider more training data or architecture changes.');
    } else if (pairwiseAUC < 0.72) {
      recommendations.push('Pairwise AUC approaching target. Fine-tune CLIP layers for better performance.');
    } else {
      recommendations.push('Excellent pairwise ranking performance! Model ready for production.');
    }
    
    if (ctrMetrics.correlation < 0.3) {
      recommendations.push('Low CTR correlation. Consider adding more engagement features.');
    } else if (ctrMetrics.correlation > 0.5) {
      recommendations.push('Strong CTR correlation. Model shows good predictive power.');
    }
    
    if (ctrMetrics.mse > 0.01) {
      recommendations.push('High CTR MSE. Consider regularization or more training epochs.');
    }
    
    return recommendations;
  }
}

// Main Training Interface
export class ThumbnailModelTraining {
  private trainer: ModelTrainer;
  private evaluator: ModelEvaluator;
  
  constructor() {
    this.trainer = new ModelTrainer();
    this.evaluator = new ModelEvaluator();
  }
  
  async runFullTrainingPipeline(): Promise<TrainingResults> {
    const config: TrainingConfig = {
      modelPreset: 'advanced',
      dataPath: './data/training',
      outputPath: './models',
      validationSplit: 0.2,
      testSplit: 0.1,
      enableEarlyStopping: true,
      earlyStoppingPatience: 10,
      enableCheckpointing: true,
      checkpointInterval: 50
    };
    
    console.log('🚀 Starting full model training pipeline...');
    
    // Train model
    const results = await this.trainer.trainModel(config);
    
    console.log('✅ Training completed successfully!');
    console.log(`📊 Final Metrics:`);
    console.log(`   Pairwise AUC: ${results.metrics.pairwiseAUC.toFixed(4)}`);
    console.log(`   CTR MSE: ${results.metrics.ctrMSE.toFixed(6)}`);
    console.log(`   CTR Correlation: ${results.metrics.ctrCorrelation.toFixed(4)}`);
    
    return results;
  }
}

export default ThumbnailModelTraining;
