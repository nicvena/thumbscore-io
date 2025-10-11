# Model Evaluation Guide - Data-Backed, Not Vibes

## Overview

This guide explains the comprehensive evaluation framework for YouTube thumbnail ranking models. All metrics are **data-backed** with statistical rigor, not based on subjective impressions.

## Three Core Metrics

### 1. Pairwise AUC on Held-Out Channels

**What it measures:** Model's ability to predict which thumbnail will outperform another on channels it has never seen during training.

**Target Performance:**
- ≥0.65 - Baseline (proxy data only)
- ≥0.72 - Target (with YouTube Analytics data)

**Methodology:**
- **Held-out channels**: 20-30% of total channels reserved for testing
- **Pairwise comparison**: Compare thumbnails from same channel within ±30 day window
- **AUC calculation**: Area Under ROC Curve for ranking decisions
- **Statistical rigor**: Bootstrap confidence intervals (1000 iterations)
- **Breakdown**: Per-channel and per-niche analysis

**Why it matters:**
- Tests generalization to new creators
- Reduces overfitting to specific channels
- Validates model works across different content styles

```typescript
// Example evaluation
const heldOutChannels = ['channel_1', 'channel_2', ...]; // 20-30% of channels
const pairwiseData = [
  {
    channelId: 'channel_1',
    thumbnailA: { embedding: [...], viewsPerHour: 1500 },
    thumbnailB: { embedding: [...], viewsPerHour: 2200 },
    winner: 'B'
  },
  // ... more pairs
];

const auc = await evaluator.evaluatePairwiseAUC(model, pairwiseData);
// Output: { overall: 0.68, confidence95: [0.64, 0.72] }
```

---

### 2. Spearman ρ Correlation with Views/Hour

**What it measures:** How well predicted scores correlate with actual video performance within each channel.

**Target Performance:**
- \>0.3 - Minimum acceptable
- \>0.5 - Good performance
- Statistical significance: p < 0.05

**Methodology:**
- **Ground truth**: views_per_hour = views / (now - publishedAt)
- **Within-channel**: Correlation computed separately per channel, then aggregated
- **Rank correlation**: Spearman ρ (robust to outliers)
- **Significance**: p-value from t-distribution
- **Normalization**: Channel baseline z-score adjustment

**Why it matters:**
- Tests prediction accuracy, not just ranking
- Within-channel comparison reduces confounders (channel size, niche)
- Validates model captures true engagement drivers

```typescript
// Example evaluation
const testData = [
  {
    channelId: 'channel_1',
    embedding: [...],
    viewsPerHour: 1500,
    predictedScore: 0.72
  },
  // ... more samples
];

const correlation = await evaluator.evaluateSpearmanCorrelation(model, testData);
// Output: { overall: 0.42, pValue: 0.001, byChannel: {...} }
```

---

### 3. A/B Test Win Rate (Phase 2 - Live Deployment)

**What it measures:** Percentage of times model-selected thumbnails outperform creator's original choice in real YouTube A/B tests.

**Target Performance:**
- \>50% - Model adds value
- \>60% - Strong performance
- \>70% - Exceptional

**Methodology:**
- **Live testing**: Deploy both thumbnails with 50/50 traffic split
- **Statistical power**: Minimum 10,000 impressions per variant
- **Winner determination**: Higher CTR after reaching significance
- **Confidence**: Wilson score interval for win rate
- **Breakdown**: Results stratified by niche, channel size, video type

**Why it matters:**
- Ultimate test: real user behavior on YouTube
- No proxy metrics - actual CTR comparison
- Validates model in production conditions

```typescript
// Example A/B test results
const abTests = [
  {
    testId: 'test_001',
    channelId: 'channel_1',
    modelThumbnail: 'model_v1.jpg',
    creatorThumbnail: 'original.jpg',
    modelCTR: 0.045,      // 4.5%
    creatorCTR: 0.038,    // 3.8%
    winner: 'model',
    impressions: 50000,
    confidence: 0.95
  },
  // ... more tests
];

const winRate = await evaluator.evaluateABTestWinRate(abTests);
// Output: { overall: 0.62, confidence95: [0.54, 0.70] }
```

---

## Evaluation Pipeline

### Setup

```typescript
import { ModelEvaluationPipeline } from '@/lib/model-evaluation';

const pipeline = new ModelEvaluationPipeline();

// Prepare evaluation dataset
const dataset = {
  heldOutChannels: ['channel_1', 'channel_2', ...],
  testSamples: [...],
  pairwiseSamples: [...]
};

// Run evaluation
const metrics = await pipeline.evaluateModel(model, dataset, abTestResults);
```

### Interpreting Results

```typescript
{
  pairwiseAUC: {
    overall: 0.68,
    confidence95: [0.64, 0.72],
    byChannel: Map { 'channel_1': 0.72, ... },
    byNiche: Map { 'tech': 0.71, 'gaming': 0.65, ... }
  },
  spearmanCorrelation: {
    overall: 0.42,
    pValue: 0.001,
    byChannel: Map { ... },
    byNiche: Map { ... }
  },
  abTestWinRate: {
    overall: 0.62,
    modelWins: 31,
    creatorWins: 19,
    totalTests: 50,
    confidenceInterval95: [0.54, 0.70]
  },
  diagnostics: {
    calibration: 0.87,      // How well probabilities match outcomes
    coverage: 0.85,         // Proportion of space covered
    fairness: Map { ... }   // Performance across niches
  }
}
```

---

## Data Requirements

### For Pairwise AUC:
- **Minimum**: 50 held-out channels
- **Optimal**: 100+ held-out channels
- **Pairs per channel**: 10-50 pairwise comparisons
- **Total pairs**: 1,000+ for stable AUC

### For Spearman ρ:
- **Minimum**: 5 videos per channel
- **Optimal**: 20+ videos per channel
- **Total samples**: 500+ across all channels
- **Time window**: Videos from last 90 days preferred

### For A/B Tests:
- **Minimum**: 10,000 impressions per test
- **Optimal**: 50,000+ impressions per test
- **Total tests**: 30+ for stable win rate
- **Duration**: 7-14 days per test typical

---

## Statistical Rigor

### Confidence Intervals
All metrics include 95% confidence intervals using appropriate methods:
- **AUC**: Bootstrap resampling (1000 iterations)
- **Win Rate**: Wilson score interval
- **Correlation**: Fisher's z-transformation

### Significance Testing
- **Spearman ρ**: p-value from t-distribution
- **A/B tests**: Chi-square test for CTR difference
- **Minimum p-value**: 0.05 (5% significance level)

### Multiple Hypothesis Correction
When evaluating across multiple niches, apply Bonferroni correction:
```typescript
const adjustedAlpha = 0.05 / numberOfNiches;
```

---

## Best Practices

### 1. Hold-Out Strategy
- **Never** test on training channels
- Reserve 20-30% channels for testing
- Stratify by niche and size
- Ensure representation across all niches

### 2. Temporal Validation
- Test on recent data (last 30 days)
- Avoid temporal leakage
- Monitor for drift over time

### 3. Fairness Evaluation
- Check performance across niches
- Identify underperforming segments
- Ensure no systematic bias

### 4. Continuous Monitoring
- Re-evaluate monthly with new data
- Track metric trends over time
- Set up automated evaluation pipeline

---

## API Usage

### Evaluate Model
```bash
curl -X POST http://localhost:3000/api/evaluate-model \
  -H "Content-Type: application/json" \
  -d '{
    "modelPreset": "advanced",
    "includeABTests": true,
    "generateMockData": false
  }'
```

### Response
```json
{
  "message": "Model evaluation completed",
  "metrics": {
    "pairwiseAUC": { ... },
    "spearmanCorrelation": { ... },
    "abTestWinRate": { ... }
  },
  "readiness": {
    "status": "production_ready",
    "confidence": 0.95,
    "summary": "Model exceeds all target metrics"
  },
  "recommendations": [...]
}
```

---

## Troubleshooting

### Low Pairwise AUC (<0.65)
**Possible causes:**
- Insufficient training data
- Model overfitting to training channels
- Poor feature extraction

**Solutions:**
- Increase training data to 200k+ pairs
- Add regularization (dropout, weight decay)
- Fine-tune CLIP layers (last 2-4 blocks)

### Low Spearman ρ (<0.3)
**Possible causes:**
- Weak signal in proxy metric (views_per_hour)
- Missing important features
- Poor channel baseline normalization

**Solutions:**
- Add engagement features (likes, comments)
- Use true CTR if available from Analytics
- Improve normalization strategy

### Poor A/B Win Rate (<50%)
**Possible causes:**
- Model not learning true CTR drivers
- Domain shift (training vs. production)
- Insufficient test duration

**Solutions:**
- Analyze failure cases
- Collect more live feedback data
- Increase test duration for significance

---

## Roadmap

### Phase 1: Proxy Metrics (Current)
- ✅ Pairwise AUC on views_per_hour
- ✅ Spearman correlation within channels
- ✅ Target: ≥0.65 AUC, >0.3 correlation

### Phase 2: YouTube Analytics Integration
- 🔄 Access to true CTR data via Analytics API
- 🔄 Gold-label pairs for supervised learning
- 🔄 Target: ≥0.72 AUC, >0.5 correlation

### Phase 3: Live A/B Testing
- 📋 Deploy model recommendations
- 📋 50/50 traffic split tests
- 📋 Target: >50% win rate vs. creator choice

### Phase 4: Continuous Learning
- 📋 Feedback loop from live tests
- 📋 Automated retraining pipeline
- 📋 Online learning from new data

---

## References

1. **Pairwise Ranking Loss**: Learning to Rank using Gradient Descent (Burges et al., 2005)
2. **AUC Optimization**: The Area Under the ROC Curve (Hanley & McNeil, 1982)
3. **Spearman Correlation**: The Proof and Measurement of Association (Spearman, 1904)
4. **A/B Testing**: Trustworthy Online Controlled Experiments (Kohavi et al., 2020)
5. **Bootstrap CI**: Bootstrap Methods (Efron & Tibshirani, 1994)

---

## Support

For questions or issues with evaluation:
- Check console logs for detailed diagnostics
- Review confidence intervals for stability
- Compare across niches for fairness
- Contact ML team for interpretation help

**Remember: Data-backed evaluation, not vibes!** 📊
