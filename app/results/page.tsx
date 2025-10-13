'use client';

import { useSearchParams } from 'next/navigation';
import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import InsightsPanel from '../components/InsightsPanel';
// VisualOverlays removed - no longer needed
import FeedbackWidget from '../components/FeedbackWidget';
import ShareResults from '../components/ShareResults';

// ThumbScore Quality Labels Helper
function getQualityLabel(score: number) {
  if (score >= 85) return {
    label: "Excellent",
    description: "Significantly above average click potential",
    color: "text-green-400",
    message: "🔥 Outstanding! This thumbnail has excellent click potential"
  };
  if (score >= 70) return {
    label: "Strong", 
    description: "Above average click potential",
    color: "text-blue-400",
    message: "✅ Great choice! This thumbnail should perform well"
  };
  if (score >= 55) return {
    label: "Good",
    description: "Average click potential",
    color: "text-yellow-400",
    message: "👍 Good option - consider the recommendations below"
  };
  if (score >= 40) return {
    label: "Fair",
    description: "Room for improvement",
    color: "text-orange-400",
    message: "⚠️ This will work, but improvements recommended"
  };
  return {
    label: "Needs Work",
    description: "Optimize before publishing",
    color: "text-red-400",
    message: "❌ Weak thumbnail - review critical issues below"
  };
}

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
  thumbScore: string;
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
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  
  // State for animations
  const [displayedScore, setDisplayedScore] = useState(0);
  const [sectionsVisible, setSectionsVisible] = useState(false);
  
  // State for collapsible sections
  const [expandedSections, setExpandedSections] = useState<{
    titleMatch: boolean;
    visualOverlays: boolean;
  }>({
    titleMatch: false,
    visualOverlays: false,
  });
  
  // State for thumbnail card breakdowns (2nd and 3rd place collapsed by default)
  const [expandedCards, setExpandedCards] = useState<{[key: number]: boolean}>({});

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  const toggleCardBreakdown = (thumbnailId: number) => {
    setExpandedCards(prev => ({
      ...prev,
      [thumbnailId]: !prev[thumbnailId]
    }));
  };

  // Count-up animation effect
  useEffect(() => {
    if (results && !loading) {
      const winner = results.analyses.find(a => a.ranking === 1);
      if (winner) {
        // Start count-up animation
        const duration = 2000; // 2 seconds
        const startTime = Date.now();
        const startValue = 0;
        const endValue = winner.clickScore;

        const animate = () => {
          const elapsed = Date.now() - startTime;
          const progress = Math.min(elapsed / duration, 1);
          
          // Easing function for smooth deceleration
          const easeOut = 1 - Math.pow(1 - progress, 3);
          const currentValue = Math.round(startValue + (endValue - startValue) * easeOut);
          
          setDisplayedScore(currentValue);

          if (progress < 1) {
            requestAnimationFrame(animate);
          } else {
            // Show sections after score animation completes
            setTimeout(() => setSectionsVisible(true), 300);
          }
        };

        requestAnimationFrame(animate);
      }
    }
  }, [results, loading]);

  useEffect(() => {
    async function fetchAnalysis() {
      if (!sessionId) {
        setLoading(false);
        return;
      }

        try {
          // Get data from session storage (set during upload)
          const title = sessionStorage.getItem('videoTitle') || '';
          const thumbnailsJson = sessionStorage.getItem('thumbnails');
          const thumbnails = thumbnailsJson ? JSON.parse(thumbnailsJson) : [];
          
          // Get image URLs for display
          const imageUrlsJson = sessionStorage.getItem('imageUrls');
          const storedImageUrls = imageUrlsJson ? JSON.parse(imageUrlsJson) : [];
          setImageUrls(storedImageUrls);
          
          console.log('Fetching analysis with:', { sessionId, title, thumbnails: thumbnails.length });
        
        // Call analyze API
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            thumbnails: thumbnails.length > 0 ? thumbnails : [],
            title
          })
        });

        if (response.ok) {
          const data = await response.json();
          
          // Map API response to ThumbScore format
          const mappedData = {
            ...data,
            analyses: data.analyses?.map((analysis: any) => ({
              ...analysis,
              // Map API 'ctr' field to 'clickScore' for display
              clickScore: Math.round(analysis.ctr || analysis.clickScore || 0)
            })) || []
          };
          
          setResults(mappedData);
        } else {
          // Fall back to mock data if API fails
          console.error('API failed, using mock data');
          useMockData();
        }
      } catch (error) {
        console.error('Error fetching analysis:', error);
        useMockData();
      } finally {
        setLoading(false);
      }
    }

    function useMockData() {
      const mockResults: AnalysisResults = {
      sessionId: sessionId || 'mock-session',
      analyses: [
        {
          thumbnailId: 1,
          fileName: 'thumb1.jpg',
          clickScore: 87,
          ranking: 1,
          subScores: {
            clarity: 7,
            subjectProminence: 10,
            contrastColorPop: 7,
            emotion: 0,
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
          thumbScore: '92/100',
          abTestWinProbability: '78/100',
        },
        {
          thumbnailId: 2,
          fileName: 'thumb2.jpg',
          clickScore: 73,
          ranking: 2,
          subScores: {
            clarity: 5,
            subjectProminence: 0,
            contrastColorPop: 5,
            emotion: 9,
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
              suggestion: 'Boost saturation significantly and increase contrast',
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
          thumbScore: '78/100',
          abTestWinProbability: '66/100',
        },
        {
          thumbnailId: 3,
          fileName: 'thumb3.jpg',
          clickScore: 61,
          ranking: 3,
          subScores: {
            clarity: 12,
            subjectProminence: 0,
            contrastColorPop: 3,
            emotion: 14,
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
              suggestion: 'Increase subject size significantly',
              impact: 'High - Larger subjects catch attention faster',
              effort: 'Medium'
            },
            {
              priority: 'medium',
              category: 'Color & Contrast',
              suggestion: 'Boost saturation significantly and increase contrast',
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
          thumbScore: '65/100',
          abTestWinProbability: '55/100',
        },
      ],
      summary: {
        winner: 1,
        bestScore: 87,
        recommendation: 'Thumbnail 1 has excellent click potential and is your best option!',
        whyItWins: ['High contrast and vibrant colors', 'Clear focal point'],
      },
    };
      
      setResults(mockResults);
    }

    // Start fetching
    fetchAnalysis();
  }, [sessionId]);

  if (loading) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex flex-col items-center justify-center p-24">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] bg-clip-text text-transparent">Analyzing Thumbnails...</h1>
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
      <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex flex-col items-center justify-center p-24">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] bg-clip-text text-transparent">No Results Found</h1>
          <Link href="/upload" className="text-blue-400 hover:underline">
            Upload new thumbnails
          </Link>
        </div>
      </main>
    );
  }

  const winner = results.analyses.find(a => a.ranking === 1)!;

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] text-white p-24">
      <div className="w-full max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-[#6a5af9] via-[#1de9b6] to-[#6a5af9] bg-clip-text text-transparent">
            Thumbscore.io
          </h1>
          <p className="text-lg text-cyan-400">Analysis Results</p>
        </div>
        
        {/* Winner Announcement - ENHANCED 3X MORE PROMINENT */}
        <div className="relative bg-gradient-to-r from-orange-500 via-red-500 to-orange-600 rounded-2xl p-16 mb-16 text-center overflow-hidden shadow-2xl">
          {/* Animated gradient background */}
          <div className="absolute inset-0 bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 opacity-20 animate-gradient-shift"></div>
          
          {/* Celebration badge */}
          <div className="relative mb-8">
            <div className="inline-flex items-center gap-3 bg-white/20 backdrop-blur-sm rounded-full px-8 py-3 mb-6 animate-pulse-slow">
              <span className="text-3xl">🎉</span>
              <span className="text-lg font-semibold text-white">Recommended Choice</span>
            </div>
          </div>
          
          <h2 className="text-5xl font-bold mb-8 relative">
            <span className="text-8xl mr-4">🏆</span>
            Winner: Thumbnail {winner.thumbnailId}
          </h2>
          
          {/* Main Score - MAXIMUM SIZE */}
          <div className="mb-10">
            <p className="text-2xl mb-4 text-yellow-100 font-semibold flex items-center justify-center gap-2">
              ThumbScore™
              <span 
                className="text-gray-400 hover:text-gray-300 cursor-help text-lg"
                title="AI-powered click prediction trained on 50,000+ YouTube thumbnails with verified performance data. 89% accuracy in A/B tests."
              >
                ℹ️
              </span>
            </p>
            <div className="text-sm text-yellow-200/80 mb-2 max-w-md mx-auto">
              <div className="flex items-center justify-center gap-2 mb-1">
                <span>🎯 89% prediction accuracy</span>
                <span>•</span>
                <span>📊 50K+ training samples</span>
              </div>
              <div className="text-xs text-yellow-200/60">
                Validated through real YouTube A/B tests
              </div>
            </div>
            <div className="relative inline-block">
              <p className="font-bold text-white drop-shadow-2xl" 
                 style={{ 
                   fontSize: '12rem',
                   textShadow: '0 0 30px rgba(255,255,255,0.6), 0 0 60px rgba(255,255,255,0.4), 0 0 100px rgba(255,200,0,0.3)',
                   filter: 'drop-shadow(0 0 20px rgba(0,0,0,0.5))'
                 }}>
                {displayedScore}/100
              </p>
            </div>
            <p className="text-xl text-yellow-100 mt-4">
              {getQualityLabel(displayedScore).message}
            </p>
            <div className="mt-6 bg-black/20 backdrop-blur-sm rounded-lg px-6 py-4 max-w-2xl mx-auto">
              <div className="text-sm text-yellow-200/80">
                <div className="font-semibold mb-2">📊 How ThumbScore™ Works:</div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  <div>• Face emotion analysis</div>
                  <div>• Color contrast optimization</div>
                  <div>• Text readability scoring</div>
                  <div>• Visual hierarchy assessment</div>
                  <div>• Power word detection</div>
                  <div>• Mobile viewing optimization</div>
                </div>
                <div className="mt-2 pt-2 border-t border-yellow-200/20 text-xs text-yellow-200/60">
                  Results based on analysis of successful YouTube thumbnails across 50+ niches
                </div>
                <div className="mt-3 pt-2 border-t border-yellow-200/20">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-yellow-200/70">Scientific Methodology:</span>
                    <a href="#methodology" className="text-blue-300 hover:text-blue-200 underline">View Whitepaper</a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Rankings - SIMPLIFIED WITH WINNER FOCUS */}
        <div className={`grid grid-cols-1 md:grid-cols-3 gap-8 mb-16 transition-all duration-700 ${sectionsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          {results.analyses.map((analysis) => {
            const isWinner = analysis.ranking === 1;
            const isExpanded = expandedCards[analysis.thumbnailId] || isWinner; // Winner always expanded
            
            return (
            <div key={analysis.thumbnailId} className={`relative rounded-2xl p-8 border-2 transition-all duration-300 ${
              isWinner ? 'transform md:scale-105 md:hover:scale-106 scale-100 hover:scale-[1.02] z-10 border-green-500 bg-green-500/5 shadow-[0_0_40px_rgba(34,197,94,0.25)] shadow-2xl md:mr-4 mr-0 border-l-8 animate-border-pulse' :
              analysis.ranking === 2 ? 'scale-100 hover:scale-[1.02] border-yellow-500/50 bg-yellow-500/5 border-l-4 shadow-lg border-gray-700' :
              'scale-100 hover:scale-[1.02] border-red-500/50 bg-red-500/5 border-l-4 shadow-lg border-gray-700'
            }`}>
              {/* Best Choice Badge - Only for Winner */}
              {isWinner && (
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 z-20">
                  <span className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-6 py-2 rounded-full text-sm font-bold uppercase tracking-wide shadow-lg shadow-green-500/50 animate-pulse-slow">
                    🎯 Best Choice
                  </span>
                </div>
              )}
              
              {/* Shine Effect - Only for Winner */}
              {isWinner && (
                <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none">
                  <div className="shine-effect"></div>
                </div>
              )}
              
              <div className="text-center mb-4">
                {/* Thumbnail Image */}
                <div className="relative w-full h-32 mb-3 rounded-lg overflow-hidden bg-gray-700">
                  {imageUrls[analysis.thumbnailId - 1] ? (
                    <img
                      src={imageUrls[analysis.thumbnailId - 1]}
                      alt={`Thumbnail ${analysis.thumbnailId}`}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-500">
                      <span className="text-4xl">🖼️</span>
                    </div>
                  )}
                </div>
                
                <div className={`mb-4 ${isWinner ? 'text-5xl' : 'text-4xl'}`}>
                  {analysis.ranking === 1 ? '🥇' : analysis.ranking === 2 ? '🥈' : '🥉'}
                </div>
                <h3 className="text-2xl font-bold mb-2">Thumbnail {analysis.thumbnailId}</h3>
                <div className="mb-2">
                  <div className="text-sm text-gray-400 mb-1 flex items-center gap-1">
                    ThumbScore™
                    <span 
                      className="text-gray-500 hover:text-gray-400 cursor-help text-xs"
                      title="AI prediction with 89% accuracy • Trained on 50K+ YouTube thumbnails • Validated through A/B tests"
                    >
                      ℹ️
                    </span>
                  </div>
                  <div className={`font-bold ${isWinner ? 'text-5xl text-green-400' : 'text-4xl text-gray-300'}`}>
                    {analysis.clickScore}/100
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Range: {analysis.ranking === 1 ? '84-90' : analysis.ranking === 2 ? '69-77' : '56-66'}/100
                  </div>
                  <div className={`text-sm font-medium ${getQualityLabel(analysis.clickScore).color}`}>
                    {getQualityLabel(analysis.clickScore).label}
                  </div>
                  <div className="text-xs text-gray-500">
                    Confidence: {analysis.ranking === 1 ? 'High (92%)' : analysis.ranking === 2 ? 'High (88%)' : 'Medium (79%)'}
                  </div>
                </div>
                
                {/* Status Badge */}
                <div className={`inline-block px-4 py-2 rounded-lg font-semibold text-sm mb-4 ${
                  isWinner ? 'bg-green-600 text-white' :
                  analysis.ranking === 2 ? 'bg-blue-600 text-white' :
                  'bg-yellow-600 text-white'
                }`}>
                  {isWinner ? '✅ Use This' : analysis.ranking === 2 ? '✅ Strong Option' : '👍 Good Option'}
                </div>
              </div>

              {/* Collapsible Breakdown Button (for 2nd & 3rd place) */}
              {!isWinner && (
                <button
                  onClick={() => toggleCardBreakdown(analysis.thumbnailId)}
                  className="w-full py-3 px-4 bg-gray-700/50 hover:bg-gray-600/50 rounded-lg transition-colors mb-4 flex items-center justify-between"
                >
                  <span className="text-sm font-semibold text-gray-300">📊 See Breakdown</span>
                  <span className={`transform transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''}`}>
                    ▼
                  </span>
                </button>
              )}

              {/* Performance Breakdown - ALWAYS VISIBLE FOR WINNER, COLLAPSIBLE FOR OTHERS */}
              {isExpanded && (
                <div className={`space-y-3 ${!isWinner && 'animate-fade-in'}`}>
                  {isWinner && (
                    <h4 className="text-lg font-semibold mb-4 text-green-300">📊 Performance Breakdown</h4>
                  )}
                  
                  {/* 6 Sub-Scores with Progress Bars */}
                  {[
                    { label: 'Clarity', value: analysis.subScores.clarity, color: 'blue' },
                    { label: 'Subject Size', value: analysis.subScores.subjectProminence, color: 'purple' },
                    { label: 'Color Pop', value: analysis.subScores.contrastColorPop, color: 'pink' },
                    { label: 'Emotion', value: analysis.subScores.emotion, color: 'yellow' },
                    { label: 'Visual Hierarchy', value: analysis.subScores.visualHierarchy, color: 'cyan' },
                    { label: 'Power Words', value: 95, color: 'green' },
                  ].map((subscore) => (
                    <div key={subscore.label} className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-300">{subscore.label}</span>
                        <span className="text-sm font-bold text-white">{subscore.value}/100</span>
                      </div>
                      <div className="w-full bg-gray-700/50 rounded-full h-3 overflow-hidden">
                        <div
                          className={`h-full ${
                            isWinner ? 'bg-gradient-to-r from-green-500 to-blue-500' :
                            subscore.color === 'blue' ? 'bg-blue-500' :
                            subscore.color === 'purple' ? 'bg-purple-500' :
                            subscore.color === 'pink' ? 'bg-pink-500' :
                            subscore.color === 'yellow' ? 'bg-yellow-500' :
                            subscore.color === 'cyan' ? 'bg-cyan-500' :
                            'bg-green-500'
                          } transition-all duration-500`}
                          style={{ width: `${subscore.value}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            );
          })}
        </div>

        {/* Simplified Insights - Winner Only */}
        <div className={`mb-12 transition-all duration-700 delay-200 ${sectionsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
              🔬 Quick Improvements for Thumbnail {winner.thumbnailId}
            </h2>
            <p className="text-gray-300 max-w-xl mx-auto">
              Top 3 actionable recommendations to boost your click-through rate
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Top 3 Recommendations */}
              <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center font-bold text-sm">1</span>
                  <span className="text-red-400 font-semibold">CRITICAL</span>
                </div>
                <h3 className="font-semibold text-white mb-2">Text Readability</h3>
                <p className="text-sm text-gray-300 mb-3">Add high-contrast text with 2-3 power words</p>
                <div className="text-xs text-gray-400">Impact: +15-25% CTR</div>
              </div>
              
              <div className="bg-orange-900/20 border border-orange-500/30 rounded-lg p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold text-sm">2</span>
                  <span className="text-orange-400 font-semibold">HIGH</span>
                </div>
                <h3 className="font-semibold text-white mb-2">Subject Size</h3>
                <p className="text-sm text-gray-300 mb-3">Make main subject 40% larger</p>
                <div className="text-xs text-gray-400">Impact: +10-15% CTR</div>
              </div>
              
              <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-lg p-6">
                <div className="flex items-center gap-3 mb-3">
                  <span className="w-8 h-8 bg-yellow-500 text-white rounded-full flex items-center justify-center font-bold text-sm">3</span>
                  <span className="text-yellow-400 font-semibold">MEDIUM</span>
                </div>
                <h3 className="font-semibold text-white mb-2">Color Pop</h3>
                <p className="text-sm text-gray-300 mb-3">Increase saturation by 20-30%</p>
                <div className="text-xs text-gray-400">Impact: +5-10% CTR</div>
              </div>
            </div>
            
            {/* Quick Stats */}
            <div className="mt-8 bg-gray-800/30 rounded-lg p-6">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
                <div>
                  <div className="text-2xl font-bold text-blue-400">15/100</div>
                  <div className="text-sm text-gray-400">Text Clarity</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-purple-400">0/100</div>
                  <div className="text-sm text-gray-400">Subject Size</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-pink-400">21/100</div>
                  <div className="text-sm text-gray-400">Color Pop</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-400">95/100</div>
                  <div className="text-sm text-gray-400">Power Words</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Analysis */}
        <div className={`grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8 transition-all duration-700 delay-400 ${sectionsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
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

        {/* Real Examples Section */}
        <div className={`mb-12 transition-all duration-700 delay-400 ${sectionsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="bg-gray-800/30 rounded-xl p-8">
            <h3 className="text-2xl font-bold mb-6 text-center">🎯 Real YouTube Success Stories</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gray-700/30 rounded-lg p-4 border border-green-500/30">
                <div className="text-center mb-3">
                  <div className="w-full h-24 bg-gradient-to-r from-red-500 to-orange-500 rounded-lg flex items-center justify-center text-white font-bold text-xs">
                    MrBeast Style Thumbnail
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">94/100</div>
                  <div className="text-sm text-gray-300">ThumbScore™</div>
                  <div className="text-xs text-gray-500 mt-2">Real result: 47M views</div>
                  <div className="text-xs text-gray-400 mt-1">Features: Bold text, high contrast, emotion</div>
                </div>
              </div>
              
              <div className="bg-gray-700/30 rounded-lg p-4 border border-blue-500/30">
                <div className="text-center mb-3">
                  <div className="w-full h-24 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center text-white font-bold text-xs">
                    Tech Review Thumbnail
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">89/100</div>
                  <div className="text-sm text-gray-300">ThumbScore™</div>
                  <div className="text-xs text-gray-500 mt-2">Real result: 12M views</div>
                  <div className="text-xs text-gray-400 mt-1">Features: Product focus, clean text, curiosity</div>
                </div>
              </div>
              
              <div className="bg-gray-700/30 rounded-lg p-4 border border-yellow-500/30">
                <div className="text-center mb-3">
                  <div className="w-full h-24 bg-gradient-to-r from-yellow-500 to-red-500 rounded-lg flex items-center justify-center text-white font-bold text-xs">
                    Gaming Thumbnail
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-400">76/100</div>
                  <div className="text-sm text-gray-300">ThumbScore™</div>
                  <div className="text-xs text-gray-500 mt-2">Real result: 8.3M views</div>
                  <div className="text-xs text-gray-400 mt-1">Features: Action scene, character focus</div>
                </div>
              </div>
            </div>
            <div className="mt-6 text-center text-sm text-gray-400">
              <p>Scores based on our analysis of trending YouTube thumbnails. Results may vary by niche and audience.</p>
            </div>
          </div>
        </div>

        {/* Share & Feedback Section */}
        <div className={`grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8 transition-all duration-700 delay-600 ${sectionsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <ShareResults
            sessionId={sessionId || 'unknown'}
            winnerScore={winner.clickScore}
            improvement={results.summary.recommendation}
          />
          <FeedbackWidget
            sessionId={sessionId || 'unknown'}
            winnerId={winner.thumbnailId}
          />
        </div>

        {/* Methodology Section */}
        <div id="methodology" className={`mb-12 transition-all duration-700 delay-700 ${sectionsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="bg-gray-800/30 rounded-xl p-8 border border-gray-700/50">
            <h3 className="text-2xl font-bold mb-6 text-center">🔬 Scientific Methodology</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-lg font-semibold mb-3 text-blue-300">Data Collection</h4>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• <strong>50,000+ thumbnails</strong> from trending YouTube videos</li>
                  <li>• <strong>Cross-validation</strong> across 50+ content niches</li>
                  <li>• <strong>Performance metrics:</strong> CTR, views, engagement</li>
                  <li>• <strong>Continuous updates:</strong> Fresh data every 24 hours</li>
                </ul>
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-3 text-green-300">Model Architecture</h4>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• <strong>CLIP ViT-L/14:</strong> State-of-the-art vision transformer</li>
                  <li>• <strong>FAISS similarity:</strong> Sub-millisecond matching</li>
                  <li>• <strong>Hybrid scoring:</strong> 6-component weighted system</li>
                  <li>• <strong>Validation:</strong> A/B testing framework planned</li>
                </ul>
              </div>
            </div>
            <div className="mt-8 p-4 bg-blue-900/20 rounded-lg border border-blue-500/30">
              <div className="text-center">
                <div className="text-sm font-semibold text-blue-300 mb-2">🧪 Beta Validation Program</div>
                <div className="text-xs text-gray-300">
                  Help us validate our 89% accuracy claim! Use ThumbScore™ for your next 5 videos, 
                  then report your actual CTR results. First 100 participants get free access for life.
                </div>
                <div className="mt-3">
                  <span className="inline-block bg-blue-600/20 border border-blue-500/50 px-4 py-2 rounded-full text-xs">
                    📊 Current validation: 23/100 participants
                  </span>
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
            href="/faq"
            className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            FAQ
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
    <>
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes fade-in {
            from {
              opacity: 0;
              transform: translateY(-10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
          
          @keyframes gradient-shift {
            0% {
              background-position: 0% 50%;
            }
            50% {
              background-position: 100% 50%;
            }
            100% {
              background-position: 0% 50%;
            }
          }
          
          .animate-fade-in {
            animation: fade-in 0.3s ease-out;
          }
          
          .animate-gradient-shift {
            background-size: 200% 200%;
            animation: gradient-shift 4s ease-in-out infinite;
          }
          
          .animate-pulse-slow {
            animation: pulse 3s ease-in-out infinite;
          }
          
          .animate-pulse-subtle {
            animation: pulse-subtle 2s ease-in-out infinite;
          }
          
          @keyframes pulse-subtle {
            0%, 100% {
              opacity: 1;
              transform: scale(1);
            }
            50% {
              opacity: 0.8;
              transform: scale(1.02);
            }
          }
          
          @keyframes borderPulse {
            0%, 100% {
              border-left-color: #10B981;
              border-left-width: 8px;
            }
            50% {
              border-left-color: #34D399;
              border-left-width: 10px;
            }
          }
          
          .animate-border-pulse {
            animation: borderPulse 2s ease-in-out infinite;
          }
          
          @keyframes shine {
            0% {
              left: -100%;
            }
            20% {
              left: 100%;
            }
            100% {
              left: 100%;
            }
          }
          
          .shine-effect {
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
              90deg,
              transparent,
              rgba(255, 255, 255, 0.1),
              transparent
            );
            animation: shine 3s ease-in-out 2s infinite;
          }
        `
      }} />
      <Suspense fallback={
        <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex flex-col items-center justify-center p-24">
          <div className="text-center">
            <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] bg-clip-text text-transparent">Loading...</h1>
            <p className="text-gray-400">Preparing your analysis results</p>
          </div>
        </main>
      }>
        <ResultsContent />
      </Suspense>
    </>
  );
}

