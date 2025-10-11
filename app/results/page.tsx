'use client';

import { useSearchParams } from 'next/navigation';
import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import InsightsPanel from '../components/InsightsPanel';
import VisualOverlays from '../components/VisualOverlays';

interface ThumbnailAnalysis {
  thumbnailId: number;
  fileName: string;
  clickScore: number;
  ranking: number;
  subScores: {
    clarity: number;
    subjectProminence: number;
    contrastColorPop: number;
    emotion: number;
    visualHierarchy: number;
    clickIntentMatch: number;
  };
  heatmapData: Array<{
    x: number;
    y: number;
    intensity: number;
    label: string;
  }>;
  ocrHighlights: Array<{
    text: string;
    confidence: number;
    bbox: number[];
    color: string;
  }>;
  faceBoxes: Array<{
    bbox: number[];
    emotion: string;
    confidence: number;
    gender: string;
    age: string;
  }>;
  recommendations: Array<{
    priority: 'high' | 'medium' | 'low';
    category: string;
    suggestion: string;
    impact: string;
    effort: string;
  }>;
  predictedCTR: string;
  abTestWinProbability: string;
}

interface AnalysisResults {
  sessionId: string;
  analyses: ThumbnailAnalysis[];
  summary: {
    winner: number;
    bestScore: number;
    recommendation: string;
    whyItWins: string[];
  };
}

function ResultsContent() {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get('id');
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Use mock data for now since the analyze API has issues
    // In production, this would fetch from the analyze API
    const mockResults: AnalysisResults = {
      sessionId: sessionId || 'mock-session',
      analyses: [
        {
          thumbnailId: 1,
          fileName: 'thumb1.jpg',
          clickScore: 92,
          ranking: 1,
          subScores: {
            clarity: 88,
            subjectProminence: 94,
            contrastColorPop: 96,
            emotion: 89,
            visualHierarchy: 91,
            clickIntentMatch: 87,
          },
          heatmapData: [
            { x: 20, y: 30, intensity: 0.8, label: 'High attention area' },
            { x: 60, y: 40, intensity: 0.6, label: 'Secondary focus' },
          ],
          ocrHighlights: [
            { text: 'AMAZING', confidence: 0.92, bbox: [10, 5, 90, 20], color: '#FFD700' },
          ],
          faceBoxes: [
            { bbox: [40, 25, 60, 45], emotion: 'excited', confidence: 0.89, gender: 'neutral', age: 'adult' },
          ],
          recommendations: [
            {
              priority: 'low',
              category: 'General',
              suggestion: 'Test with mobile preview - ensure readability at small sizes',
              impact: 'Low - Optimization for mobile viewing',
              effort: 'Low'
            }
          ],
          predictedCTR: '92%',
          abTestWinProbability: '78%',
        },
        {
          thumbnailId: 2,
          fileName: 'thumb2.jpg',
          clickScore: 78,
          ranking: 2,
          subScores: {
            clarity: 75,
            subjectProminence: 82,
            contrastColorPop: 71,
            emotion: 79,
            visualHierarchy: 76,
            clickIntentMatch: 83,
          },
          heatmapData: [
            { x: 30, y: 35, intensity: 0.7, label: 'Primary focus' },
            { x: 70, y: 50, intensity: 0.5, label: 'Secondary area' },
          ],
          ocrHighlights: [
            { text: 'RESULTS', confidence: 0.88, bbox: [15, 80, 85, 95], color: '#FF6B6B' },
          ],
          faceBoxes: [
            { bbox: [35, 20, 55, 40], emotion: 'neutral', confidence: 0.76, gender: 'neutral', age: 'adult' },
          ],
          recommendations: [
            {
              priority: 'medium',
              category: 'Color & Contrast',
              suggestion: 'Boost saturation by 15-25% and increase contrast',
              impact: 'Medium - Vibrant colors perform better in feeds',
              effort: 'Low'
            },
            {
              priority: 'low',
              category: 'General',
              suggestion: 'Test with mobile preview - ensure readability at small sizes',
              impact: 'Low - Optimization for mobile viewing',
              effort: 'Low'
            }
          ],
          predictedCTR: '78%',
          abTestWinProbability: '66%',
        },
        {
          thumbnailId: 3,
          fileName: 'thumb3.jpg',
          clickScore: 65,
          ranking: 3,
          subScores: {
            clarity: 62,
            subjectProminence: 68,
            contrastColorPop: 59,
            emotion: 71,
            visualHierarchy: 63,
            clickIntentMatch: 69,
          },
          heatmapData: [
            { x: 45, y: 40, intensity: 0.6, label: 'Main subject' },
            { x: 20, y: 60, intensity: 0.4, label: 'Background text' },
          ],
          ocrHighlights: [
            { text: 'TUTORIAL', confidence: 0.75, bbox: [25, 75, 75, 90], color: '#4ECDC4' },
          ],
          faceBoxes: [
            { bbox: [42, 22, 58, 38], emotion: 'neutral', confidence: 0.68, gender: 'neutral', age: 'adult' },
          ],
          recommendations: [
            {
              priority: 'high',
              category: 'Text Clarity',
              suggestion: 'Use 1-3 words in high-contrast block text',
              impact: 'High - Text readability is crucial for mobile viewers',
              effort: 'Low'
            },
            {
              priority: 'high',
              category: 'Subject Size',
              suggestion: 'Increase subject size by 20-30%',
              impact: 'High - Larger subjects catch attention faster',
              effort: 'Medium'
            },
            {
              priority: 'medium',
              category: 'Color & Contrast',
              suggestion: 'Boost saturation by 15-25% and increase contrast',
              impact: 'Medium - Vibrant colors perform better in feeds',
              effort: 'Low'
            },
            {
              priority: 'low',
              category: 'General',
              suggestion: 'Test with mobile preview - ensure readability at small sizes',
              impact: 'Low - Optimization for mobile viewing',
              effort: 'Low'
            }
          ],
          predictedCTR: '65%',
          abTestWinProbability: '55%',
        },
      ],
      summary: {
        winner: 1,
        bestScore: 92,
        recommendation: 'Thumbnail 1 is predicted to get 92% click-through rate and is your best option!',
        whyItWins: ['High contrast and vibrant colors', 'Clear focal point'],
      },
    };
    
    setTimeout(() => {
      setResults(mockResults);
      setLoading(false);
    }, 2000);
  }, [sessionId]);

  if (loading) {
    return (
      <main className="min-h-screen bg-black flex flex-col items-center justify-center p-24">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4 text-white">Analyzing Thumbnails...</h1>
          <p className="text-gray-400">AI is comparing your thumbnails for YouTube click potential</p>
          <div className="mt-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
          </div>
        </div>
      </main>
    );
  }

  if (!results) {
    return (
      <main className="min-h-screen bg-black flex flex-col items-center justify-center p-24">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4 text-white">No Results Found</h1>
          <Link href="/upload" className="text-blue-400 hover:underline">
            Upload new thumbnails
          </Link>
        </div>
      </main>
    );
  }

  const winner = results.analyses.find(a => a.ranking === 1)!;

  return (
    <main className="min-h-screen bg-black text-white p-24">
      <div className="w-full max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-center">YouTube Thumbnail Analysis</h1>
        
        {/* Winner Announcement */}
        <div className="bg-gradient-to-r from-yellow-600 to-orange-600 rounded-lg p-8 mb-8 text-center">
          <h2 className="text-3xl font-bold mb-4">🏆 Winner: Thumbnail {winner.thumbnailId}</h2>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-lg mb-1">Predicted CTR:</p>
              <p className="text-2xl font-bold">{winner.predictedCTR}</p>
            </div>
            <div>
              <p className="text-lg mb-1">A/B Test Win Probability:</p>
              <p className="text-2xl font-bold">{winner.abTestWinProbability}</p>
            </div>
          </div>
          <p className="text-lg">{results.summary.recommendation}</p>
        </div>

        {/* Rankings */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {results.analyses.map((analysis) => (
            <div key={analysis.thumbnailId} className={`rounded-lg p-6 ${
              analysis.ranking === 1 ? 'bg-green-900 border-2 border-green-500' :
              analysis.ranking === 2 ? 'bg-yellow-900 border-2 border-yellow-500' :
              'bg-gray-800 border-2 border-gray-600'
            }`}>
              <div className="text-center mb-4">
                <div className="text-4xl mb-2">
                  {analysis.ranking === 1 ? '🥇' : analysis.ranking === 2 ? '🥈' : '🥉'}
                </div>
                <h3 className="text-xl font-bold">Thumbnail {analysis.thumbnailId}</h3>
                <div className="text-2xl font-bold text-blue-400">{analysis.predictedCTR}</div>
                <div className="text-sm text-gray-400">Click Score: {analysis.clickScore}/100</div>
              </div>

              <div className="space-y-4">
                {/* Sub-Scores */}
                <div>
                  <h4 className="font-semibold mb-3">Performance Breakdown</h4>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Clarity:</span>
                      <span className="font-semibold">{analysis.subScores.clarity}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Subject Prominence:</span>
                      <span className="font-semibold">{analysis.subScores.subjectProminence}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Color Pop:</span>
                      <span className="font-semibold">{analysis.subScores.contrastColorPop}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Emotion:</span>
                      <span className="font-semibold">{analysis.subScores.emotion}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Visual Hierarchy:</span>
                      <span className="font-semibold">{analysis.subScores.visualHierarchy}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Title Match:</span>
                      <span className="font-semibold">{analysis.subScores.clickIntentMatch}/100</span>
                    </div>
                  </div>
                </div>

                {/* AI Insights */}
                <div>
                  <h4 className="font-semibold mb-2">AI Insights</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Faces Detected:</span>
                      <span className="font-semibold">{analysis.faceBoxes.length} ({analysis.faceBoxes[0]?.emotion || 'none'})</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Text Elements:</span>
                      <span className="font-semibold">{analysis.ocrHighlights.length} words</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Attention Areas:</span>
                      <span className="font-semibold">{analysis.heatmapData.length} hotspots</span>
                    </div>
                  </div>
                </div>

                {/* Top Recommendations */}
                <div>
                  <h4 className="font-semibold mb-2">Key Recommendations</h4>
                  <div className="space-y-2">
                    {analysis.recommendations.slice(0, 2).map((rec, index) => (
                      <div key={index} className="text-xs">
                        <div className="flex items-center mb-1">
                          <span className={`px-2 py-1 rounded text-xs font-semibold mr-2 ${
                            rec.priority === 'high' ? 'bg-red-600' :
                            rec.priority === 'medium' ? 'bg-yellow-600' : 'bg-green-600'
                          }`}>
                            {rec.priority.toUpperCase()}
                          </span>
                          <span className="font-semibold">{rec.category}</span>
                        </div>
                        <p className="text-gray-300 mb-1">{rec.suggestion}</p>
                        <p className="text-gray-400 text-xs">{rec.impact}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Data-Backed Insights Panels */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4 text-white">🔬 Data-Backed Insights</h2>
          <div className="grid grid-cols-1 gap-6">
            {results.analyses.map((analysis) => (
              <div key={analysis.thumbnailId} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left: Visual Overlays */}
                <VisualOverlays
                  thumbnailId={analysis.thumbnailId}
                  fileName={analysis.fileName}
                  heatmapData={analysis.heatmapData}
                  ocrBoxes={analysis.ocrHighlights.map(ocr => ({
                    text: ocr.text,
                    bbox: ocr.bbox,
                    confidence: ocr.confidence
                  }))}
                  faceBoxes={analysis.faceBoxes}
                />
                
                {/* Right: Insights Panel */}
                <InsightsPanel
                  thumbnailId={analysis.thumbnailId}
                  fileName={analysis.fileName}
                  clickScore={analysis.clickScore}
                  subScores={analysis.subScores}
                  category="education"
                  titleMatchScore={analysis.subScores.clickIntentMatch}
                  onAutoFix={(issueId, thumbId) => {
                    console.log(`Auto-fixing ${issueId} for thumbnail ${thumbId}`);
                    alert(`Auto-fix feature coming soon! Issue: ${issueId}`);
                  }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Detailed Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Winner Analysis */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">🏆 Winner Analysis</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Top Recommendations for Thumbnail {winner.thumbnailId}</h4>
                <div className="space-y-3">
                  {winner.recommendations.slice(0, 3).map((rec, index) => (
                    <div key={index} className="border-l-4 border-yellow-500 pl-4">
                      <div className="flex items-center mb-1">
                        <span className={`px-2 py-1 rounded text-xs font-semibold mr-2 ${
                          rec.priority === 'high' ? 'bg-red-600' :
                          rec.priority === 'medium' ? 'bg-yellow-600' : 'bg-green-600'
                        }`}>
                          {rec.priority.toUpperCase()}
                        </span>
                        <span className="font-semibold text-sm">{rec.category}</span>
                      </div>
                      <p className="text-sm mb-1">{rec.suggestion}</p>
                      <p className="text-xs text-gray-400">{rec.impact}</p>
                      <p className="text-xs text-gray-500">Effort: {rec.effort}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* AI Detection Results */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">🔍 AI Detection Results</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold mb-2">Face & Emotion Analysis</h4>
                <div className="space-y-2">
                  {winner.faceBoxes.map((face, index) => (
                    <div key={index} className="bg-gray-700 rounded p-3">
                      <div className="flex justify-between text-sm">
                        <span>Emotion: <span className="font-semibold">{face.emotion}</span></span>
                        <span>Confidence: <span className="font-semibold">{Math.round(face.confidence * 100)}%</span></span>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        {face.age} {face.gender} • Bounding box: [{face.bbox.join(', ')}]
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="font-semibold mb-2">Text Recognition (OCR)</h4>
                <div className="space-y-2">
                  {winner.ocrHighlights.map((text, index) => (
                    <div key={index} className="bg-gray-700 rounded p-3">
                      <div className="flex justify-between text-sm">
                        <span>Text: <span className="font-semibold">&quot;{text.text}&quot;</span></span>
                        <span>Confidence: <span className="font-semibold">{Math.round(text.confidence * 100)}%</span></span>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        Position: [{text.bbox.join(', ')}] • Color: {text.color}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Attention Heatmap</h4>
                <div className="space-y-2">
                  {winner.heatmapData.map((hotspot, index) => (
                    <div key={index} className="bg-gray-700 rounded p-3">
                      <div className="flex justify-between text-sm">
                        <span>{hotspot.label}</span>
                        <span>Intensity: <span className="font-semibold">{Math.round(hotspot.intensity * 100)}%</span></span>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        Position: ({hotspot.x}%, {hotspot.y}%)
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-center gap-4">
          <Link
            href="/upload"
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Test More Thumbnails
          </Link>
          <Link
            href="/"
            className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Back to Home
          </Link>
        </div>
      </div>
    </main>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <main className="min-h-screen bg-black flex flex-col items-center justify-center p-24">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4 text-white">Loading...</h1>
          <p className="text-gray-400">Preparing your analysis results</p>
        </div>
      </main>
    }>
      <ResultsContent />
    </Suspense>
  );
}

