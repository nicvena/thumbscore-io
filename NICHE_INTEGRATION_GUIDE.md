# Niche System Integration Guide

This guide shows how to properly integrate the complete niche-specific thumbnail analysis system following best practices.

## Complete Component Integration

```tsx
import { NicheBadge } from './components/NicheBadge';
import { NicheInsights } from './components/NicheInsights';
import { NicheSelector } from './components/NicheSelector';
import { useState, useEffect } from 'react';

export default function ThumbnailResults({ sessionId }) {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showNicheSelector, setShowNicheSelector] = useState(false);
  const [loading, setLoading] = useState(true);

  // Fetch analysis with niche-specific processing
  useEffect(() => {
    const fetchAnalysis = async () => {
      const title = sessionStorage.getItem('videoTitle') || '';
      const niche = sessionStorage.getItem('selectedNiche') || 'general';
      const thumbnails = JSON.parse(sessionStorage.getItem('thumbnails') || '[]');
      
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          thumbnails,
          title,
          niche  // Critical: Pass niche to backend
        })
      });

      const data = await response.json();
      setAnalysisResult(data);
      setLoading(false);
    };

    fetchAnalysis();
  }, [sessionId]);

  const handleNicheChange = () => {
    // Option 1: Re-analyze with new niche
    setShowNicheSelector(true);
    
    // Option 2: Redirect to upload page
    // if (confirm('Re-analyze with different niche?')) {
    //   window.location.href = '/upload';
    // }
  };

  if (loading) return <div>Analyzing thumbnails...</div>;
  if (!analysisResult) return <div>No results found</div>;

  return (
    <div className="max-w-7xl mx-auto p-8">
      {/* 1. Niche Badge at the top */}
      <NicheBadge 
        niche={analysisResult.metadata.niche}
        onChangeNiche={handleNicheChange}
      />

      {/* 2. Overall Score - Clean, prominent */}
      <div className="text-center mb-8">
        <h1 className="text-6xl font-bold text-white mb-2">
          {analysisResult.summary.bestScore}/100
        </h1>
        {analysisResult.summary.bestScore >= 85 && (
          <p className="text-green-400">🔥 Outstanding! This thumbnail has excellent click potential</p>
        )}
        {analysisResult.summary.bestScore >= 70 && analysisResult.summary.bestScore < 85 && (
          <p className="text-blue-400">✅ Great choice! This thumbnail should perform well</p>
        )}
        {analysisResult.summary.bestScore >= 55 && analysisResult.summary.bestScore < 70 && (
          <p className="text-yellow-400">👍 Good option - consider the recommendations below</p>
        )}
        {analysisResult.summary.bestScore >= 40 && analysisResult.summary.bestScore < 55 && (
          <p className="text-orange-400">⚠️ This will work, but improvements recommended</p>
        )}
        {analysisResult.summary.bestScore < 40 && (
          <p className="text-red-400">❌ Weak thumbnail - review critical issues below</p>
        )}
      </div>

      {/* 3. Winner Announcement */}
      <div className="bg-gradient-to-r from-green-600 to-blue-600 rounded-xl p-8 mb-8 text-center">
        <h2 className="text-3xl font-bold text-white mb-4">
          🏆 Winner: Thumbnail {analysisResult.summary.winner}
        </h2>
        <p className="text-xl text-green-100">
          {analysisResult.summary.recommendation}
        </p>
      </div>

      {/* 4. Thumbnail Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
        {analysisResult.analyses.map((thumbnail, index) => (
          <div key={thumbnail.thumbnailId} className={`
            bg-slate-800 rounded-xl p-6 border-2
            ${thumbnail.ranking === 1 ? 'border-green-500 bg-green-900/20' : 'border-slate-700'}
          `}>
            {/* Thumbnail Image */}
            <div className="aspect-video bg-slate-700 rounded-lg mb-4 flex items-center justify-center">
              <span className="text-gray-400">Thumbnail {thumbnail.thumbnailId}</span>
            </div>
            
            {/* Score and Ranking */}
            <div className="text-center mb-4">
              <div className="text-3xl font-bold text-white mb-1">
                {thumbnail.clickScore}/100
              </div>
              <div className={`text-sm font-semibold ${
                thumbnail.ranking === 1 ? 'text-green-400' :
                thumbnail.ranking === 2 ? 'text-blue-400' :
                'text-yellow-400'
              }`}>
                #{thumbnail.ranking} Choice
              </div>
            </div>

            {/* Sub-scores */}
            <div className="space-y-2">
              {[
                { label: 'Clarity', value: thumbnail.subScores.clarity },
                { label: 'Emotion', value: thumbnail.subScores.emotion },
                { label: 'Color Pop', value: thumbnail.subScores.contrastColorPop },
                { label: 'Subject Size', value: thumbnail.subScores.subjectProminence }
              ].map((score) => (
                <div key={score.label} className="flex justify-between items-center">
                  <span className="text-sm text-gray-300">{score.label}</span>
                  <span className="text-sm font-semibold text-white">{score.value}/100</span>
                </div>
              ))}
            </div>

            {/* Power Words (if title provided) */}
            {thumbnail.powerWords && (
              <div className="mt-4 pt-4 border-t border-slate-700">
                <div className="text-sm text-gray-300 mb-2">Power Words Found:</div>
                <div className="flex flex-wrap gap-1">
                  {thumbnail.powerWords.foundWords.map((word, i) => (
                    <span key={i} className="px-2 py-1 bg-blue-600 text-white text-xs rounded">
                      {word}
                    </span>
                  ))}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  Score: {thumbnail.powerWords.score}/100 ({thumbnail.powerWords.tier})
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 5. Niche-Specific Insights - Dedicated Section */}
      {analysisResult.analyses[0].nicheInsights && (
        <NicheInsights insights={analysisResult.analyses[0].nicheInsights} />
      )}

      {/* 6. Detailed Recommendations */}
      <div className="bg-slate-800 rounded-xl p-6 mb-8">
        <h3 className="text-2xl font-bold text-white mb-6">
          🔧 Improvement Recommendations
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {analysisResult.analyses[0].recommendations.slice(0, 4).map((rec, index) => (
            <div key={index} className={`
              border-l-4 rounded-r-lg p-4
              ${rec.priority === 'high' ? 'border-red-500 bg-red-900/20' :
                rec.priority === 'medium' ? 'border-yellow-500 bg-yellow-900/20' :
                'border-green-500 bg-green-900/20'}
            `}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`px-2 py-1 rounded text-xs font-bold ${
                  rec.priority === 'high' ? 'bg-red-600' :
                  rec.priority === 'medium' ? 'bg-yellow-600' :
                  'bg-green-600'
                }`}>
                  {rec.priority.toUpperCase()}
                </span>
                <span className="text-sm text-gray-300">{rec.category}</span>
              </div>
              <h4 className="font-semibold text-white mb-2">{rec.suggestion}</h4>
              <p className="text-sm text-gray-300 mb-2">{rec.impact}</p>
              <p className="text-xs text-gray-400">Effort: {rec.effort}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 7. Technical Analysis Details */}
      <div className="bg-slate-800 rounded-xl p-6">
        <h3 className="text-xl font-bold text-white mb-4">
          🔍 Technical Analysis Details
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-white mb-2">ML Model Information</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Architecture: {analysisResult.metadata.mlArchitecture}</li>
              <li>• Models: {analysisResult.metadata.models.join(', ')}</li>
              <li>• Version: {analysisResult.metadata.version}</li>
              <li>• Niche: {analysisResult.metadata.niche}</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-white mb-2">Analysis Features</h4>
            <ul className="text-sm text-gray-300 space-y-1">
              <li>• Title Analysis: {analysisResult.metadata.titleProvided ? '✅' : '❌'}</li>
              <li>• Niche Optimization: {analysisResult.metadata.nicheProvided ? '✅' : '❌'}</li>
              <li>• Power Words: {analysisResult.analyses[0].powerWords ? '✅' : '❌'}</li>
              <li>• Custom Insights: {analysisResult.analyses[0].nicheInsights ? '✅' : '❌'}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
```

## Key Implementation Points

### 1. Data Flow
```typescript
Upload Page → NicheSelector → sessionStorage → API → Results → NicheBadge + Insights
```

### 2. Component Placement Order
1. **NicheBadge** - Top of page, shows analysis context
2. **Overall Score** - Prominent display with status message  
3. **Winner Card** - Highlighted winning thumbnail
4. **Thumbnail Grid** - All options with scores
5. **NicheInsights** - Dedicated section for niche advice
6. **Recommendations** - Actionable improvement tips
7. **Technical Details** - ML model and feature information

### 3. Niche Change Handling
```typescript
const handleNicheChange = () => {
  // Option 1: Show niche selector modal
  setShowNicheSelector(true);
  
  // Option 2: Redirect to upload with confirmation
  if (confirm('Re-analyze with different niche?')) {
    window.location.href = '/upload';
  }
};
```

### 4. Color-Coded Feedback
- **85-100**: Green (Excellent)
- **70-84**: Blue (Strong)  
- **55-69**: Yellow (Good)
- **40-54**: Orange (Fair)
- **0-39**: Red (Needs Work)

## Advanced Features

### Power Words Display
```tsx
{thumbnail.powerWords && (
  <div className="mt-4 pt-4 border-t border-slate-700">
    <div className="text-sm text-gray-300 mb-2">
      Power Words ({thumbnail.powerWords.niche}):
    </div>
    <div className="flex flex-wrap gap-1">
      {thumbnail.powerWords.foundWords.map((word, i) => (
        <span key={i} className="px-2 py-1 bg-blue-600 text-white text-xs rounded">
          {word}
        </span>
      ))}
    </div>
  </div>
)}
```

### Niche-Specific Insights Integration
```tsx
{analysisResult.analyses[0].nicheInsights && (
  <NicheInsights insights={analysisResult.analyses[0].nicheInsights} />
)}
```

This creates a complete, production-ready niche-specific thumbnail analysis system with optimal UX!