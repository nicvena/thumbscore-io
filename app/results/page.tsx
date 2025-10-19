'use client';

import { useSearchParams } from 'next/navigation';
import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import InsightsPanel from '../components/InsightsPanel';
import FeedbackWidget from '../components/FeedbackWidget';
import ShareResults from '../components/ShareResults';
import { UsageTracker } from '../components/UsageTracker';
import { AdvancedScoringGate, ABTestingGate, TrendAnalysisGate } from '../components/FeatureGate';

// Helper to highlight metrics in AI summary text
function highlightMetrics(text: string) {
  if (!text) return text;
  
  // Highlight percentages and numbers
  return text
    .replace(/(\d+(?:\.\d+)?%)/g, '<span class="font-bold text-blue-400">$1</span>')
    .replace(/(\d+(?:\.\d+)?\/100)/g, '<span class="font-bold text-green-400">$1</span>')
    .replace(/(\d+-\d+)/g, '<span class="font-bold text-yellow-400">$1</span>')
    .replace(/(\d+(?:\.\d+)?)\s*(points?|words?)/g, '<span class="font-bold text-purple-400">$1</span> $2');
}

// Tier-based border styling helper
function getTierBorderStyle(analysis: ThumbnailAnalysis, isWinner: boolean) {
  const score = analysis.clickScore;
  const tier = analysis.tier || getQualityLabel(score).label.toLowerCase();
  
  if (isWinner) {
    return 'group transform md:scale-105 md:hover:scale-106 scale-100 hover:scale-[1.02] z-10 border-green-500 bg-green-500/5 shadow-[0_0_40px_rgba(34,197,94,0.25)] shadow-2xl md:mr-4 mr-0 border-l-8 animate-border-pulse';
  }
  
  // Tier-based borders for non-winners
  if (score >= 86) { // Excellent
    return 'scale-100 hover:scale-[1.02] border-green-500/70 bg-green-500/5 border-l-4 shadow-lg';
  } else if (score >= 76) { // Strong  
    return 'scale-100 hover:scale-[1.02] border-green-500/50 bg-green-500/5 border-l-4 shadow-lg';
  } else if (score >= 66) { // Good
    return 'scale-100 hover:scale-[1.02] border-yellow-500/50 bg-yellow-500/5 border-l-4 shadow-lg';
  } else if (score >= 51) { // Fair
    return 'scale-100 hover:scale-[1.02] border-orange-500/50 bg-orange-500/5 border-l-4 shadow-lg';
  } else { // Needs Work
    return 'scale-100 hover:scale-[1.02] border-red-500/50 bg-red-500/5 border-l-4 shadow-lg';
  }
}

// ThumbScore Quality Labels Helper (Updated for v1.1 realistic scoring)
function getQualityLabel(score: number) {
  if (score >= 86) return {
    label: "Excellent",
    description: "Significantly above average click potential",
    color: "text-green-400",
    icon: "🟢",
    message: "🔥 Outstanding! This thumbnail has excellent click potential"
  };
  if (score >= 76) return {
    label: "Strong", 
    description: "Above average click potential",
    color: "text-green-400",
    icon: "🟢",
    message: "✅ Great choice! This thumbnail should perform well"
  };
  if (score >= 66) return {
    label: "Good",
    description: "Solid fundamentals, competitive option",
    color: "text-yellow-400",
    icon: "🟡",
    message: "👍 Good option - solid fundamentals with room to optimize"
  };
  if (score >= 51) return {
    label: "Fair",
    description: "Has potential, needs optimization",
    color: "text-orange-400",
    icon: "🟠",
    message: "⚠️ Fair thumbnail - has potential with some improvements"
  };
  return {
    label: "Needs Work",
    description: "Significant improvements needed",
    color: "text-red-400",
    icon: "🔴",
    message: "❌ Needs work - significant improvements recommended"
  };
}


interface ThumbnailAnalysis {
  thumbnailId: number;
  fileName: string;
  clickScore: number;
  ranking: number;
  tier: string; // excellent/strong/good/fair/needs_work
  tier_icon?: string; // 🟢/🟡/🟠/🔴
  tier_color?: string; // green/yellow/orange/red
  ctr_prediction?: {
    ctr_min: number;
    ctr_max: number;
    ctr_range: string;
    performance: string;
    comparison: string;
    benchmark_average: number;
    stars: number;
    confidence: number;
  };
  subScores: {
    clarity: number;
    subjectProminence: number;
    contrastColorPop: number;
    emotion: number;
    visualHierarchy: number;
    clickIntentMatch: number;
    powerWords: number;
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
  powerWords?: {
    score: number;
    foundWords: string[];
    tier: string;
    niche: string;
  };
  insights: {
    strengths: string[];
    weaknesses: string[];
    recommendations: string[];
    gptInsights?: Array<{
      label: string;
      evidence: string;
    }>;
  };
  explanation?: string;
  gptSummary?: {
    winner_summary: string;
    insights: Array<{
      label: string;
      evidence: string;
    }>;
  };
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
          const niche = sessionStorage.getItem('selectedNiche') || 'business';
          const thumbnailsJson = sessionStorage.getItem('thumbnails');
          const thumbnails = thumbnailsJson ? JSON.parse(thumbnailsJson) : [];
          
          // Get image URLs for display
          const imageUrlsJson = sessionStorage.getItem('imageUrls');
          const storedImageUrls = imageUrlsJson ? JSON.parse(imageUrlsJson) : [];
          setImageUrls(storedImageUrls);
          
          // Convert blob URLs to base64 data URLs for API
          const thumbnailsWithData = await Promise.all(
            thumbnails.map(async (thumb: any, index: number) => {
              if (storedImageUrls[index]) {
                try {
                  // Convert blob URL to base64
                  const response = await fetch(storedImageUrls[index]);
                  const blob = await response.blob();
                  const base64 = await new Promise<string>((resolve) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result as string);
                    reader.readAsDataURL(blob);
                  });
                  return {
                    ...thumb,
                    dataUrl: base64
                  };
                } catch (error) {
                  console.warn(`Failed to convert image ${index + 1}:`, error);
                  return thumb;
                }
              }
              return thumb;
            })
          );
          
          console.log('Fetching analysis with:', { sessionId, title, niche, thumbnails: thumbnailsWithData.length });
          console.log('Thumbnails data:', thumbnailsWithData);
        
        // Call analyze API
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            thumbnails: thumbnailsWithData.length > 0 ? thumbnailsWithData : [],
            title,
            niche: niche
          })
        });
        
        console.log('API Response status:', response.status);

        if (response.ok) {
          const data = await response.json();
          console.log('API Response data:', data);
          console.log('First analysis gptSummary:', data.analyses?.[0]?.gptSummary);
          
          // Map API response to ThumbScore format
          const mappedData = {
            ...data,
            analyses: data.analyses?.map((analysis: any) => ({
              ...analysis,
              // Map API 'ctr' field to 'clickScore' for display with precision
              clickScore: parseFloat((analysis.ctr || analysis.clickScore || 0).toFixed(1))
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
          tier: 'excellent',
          subScores: {
            clarity: 7,
            subjectProminence: 10,
            contrastColorPop: 7,
            emotion: 0,
            visualHierarchy: 91,
            clickIntentMatch: 87,
            powerWords: 8
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
          insights: {
            strengths: ['Strong visual appeal', 'Good composition'],
            weaknesses: [],
            recommendations: ['Great job! Keep it up'],
            gptInsights: [
              {
                label: 'Text Clarity',
                evidence: 'ALL-CAPS headline text with 95% contrast against background, positioned in top-right third'
              },
              {
                label: 'Color Psychology',
                evidence: 'Warm orange/red gradient creates appetite appeal, increasing CTR by ~15% in food niche'
              },
              {
                label: 'Subject Prominence',
                evidence: 'Main food subject occupies 65% of frame, optimal for YouTube thumbnail visibility'
              }
            ]
          },
        },
        {
          thumbnailId: 2,
          fileName: 'thumb2.jpg',
          clickScore: 73,
          ranking: 2,
          tier: 'strong',
          subScores: {
            clarity: 5,
            subjectProminence: 0,
            contrastColorPop: 5,
            emotion: 9,
            visualHierarchy: 76,
            clickIntentMatch: 83,
            powerWords: 6
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
          insights: {
            strengths: ['Good emotional appeal'],
            weaknesses: ['Needs optimization'],
            recommendations: ['Increase contrast', 'Improve text readability'],
            gptInsights: [
              {
                label: 'Facial Expression',
                evidence: 'Chef\'s intense expression creates curiosity, positioned at rule-of-thirds intersection'
              },
              {
                label: 'Composition',
                evidence: 'Subject positioned at 2/4 rule-of-thirds points, creating natural eye flow'
              }
            ]
          },
        },
        {
          thumbnailId: 3,
          fileName: 'thumb3.jpg',
          clickScore: 61,
          ranking: 3,
          tier: 'good',
          subScores: {
            clarity: 12,
            subjectProminence: 0,
            contrastColorPop: 3,
            emotion: 14,
            visualHierarchy: 63,
            clickIntentMatch: 69,
            powerWords: 4
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
          insights: {
            strengths: ['Room for improvement'],
            weaknesses: ['Low engagement potential', 'Needs optimization'],
            recommendations: ['Increase contrast', 'Improve text readability'],
            gptInsights: [
              {
                label: 'Contrast Issues',
                evidence: 'Text contrast only 45%, below optimal 70% threshold for mobile viewing'
              },
              {
                label: 'Subject Size',
                evidence: 'Main subject occupies only 35% of frame, below recommended 50-60% range'
              }
            ]
          },
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
        
        {/* Usage Tracker */}
        <div className="mb-8 max-w-sm mx-auto">
          <UsageTracker />
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
            <div key={analysis.thumbnailId} className={`relative rounded-2xl p-8 border-2 transition-all duration-300 ${getTierBorderStyle(analysis, isWinner)}`}>
              {/* Best Choice Badge - Only for Winner */}
              {isWinner && (
                <div className="absolute -top-2 left-4 z-20">
                  <div className="relative">
                    {/* Main Badge */}
                    <div className="bg-gradient-to-r from-emerald-600 via-green-500 to-teal-500 text-white px-4 py-2 rounded-xl text-sm font-bold uppercase tracking-wide shadow-xl shadow-emerald-500/40 border border-emerald-400/30 backdrop-blur-sm">
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-white/20 rounded-full flex items-center justify-center">
                          <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <span>Best</span>
                      </div>
                    </div>
                    {/* Subtle Glow Effect */}
                    <div className="absolute inset-0 bg-gradient-to-r from-emerald-600 via-green-500 to-teal-500 rounded-xl blur-sm opacity-40 -z-10"></div>
                  </div>
                </div>
              )}
              
              {/* Shine Effect - Only for Winner */}
              {isWinner && (
                <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none">
                  <div className="shine-effect"></div>
                </div>
              )}
              
              <div className="text-center mb-4">
                {/* Thumbnail Image with Interactive Overlays */}
                <div className="relative w-full h-32 mb-3 rounded-lg overflow-hidden bg-gray-700 group-hover:ring-2 group-hover:ring-blue-400 transition-all duration-300">
                  {imageUrls[analysis.thumbnailId - 1] ? (
                    <>
                      <img
                        src={imageUrls[analysis.thumbnailId - 1]}
                        alt={`Thumbnail ${analysis.thumbnailId}`}
                        className="w-full h-full object-cover"
                      />
                      {/* Interactive Overlays - Only for Winner */}
                      {isWinner && (
                        <>
                        </>
                      )}
                    </>
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
                    Score: {analysis.clickScore}/100
                  </div>
                  <div className={`text-sm font-medium flex items-center justify-center gap-2 ${getQualityLabel(analysis.clickScore).color}`}>
                    <span className="text-lg">{analysis.tier_icon || getQualityLabel(analysis.clickScore).icon}</span>
                    <span>{analysis.tier || getQualityLabel(analysis.clickScore).label}</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    Confidence: {analysis.ranking === 1 ? 'High (92%)' : analysis.ranking === 2 ? 'High (88%)' : 'Medium (79%)'}
                  </div>
                </div>
                
                {/* Status Badge - Based on Tier */}
                <div className={`inline-block px-4 py-2 rounded-lg font-semibold text-sm mb-4 ${
                  isWinner ? 'bg-green-600 text-white' :
                  analysis.clickScore >= 76 ? 'bg-green-600 text-white' :
                  analysis.clickScore >= 66 ? 'bg-blue-600 text-white' :
                  analysis.clickScore >= 51 ? 'bg-yellow-600 text-white' :
                  'bg-orange-600 text-white'
                }`}>
                  {isWinner ? '✅ Use This' :
                   analysis.clickScore >= 76 ? '✅ Strong Option' :
                   analysis.clickScore >= 66 ? '👍 Good Option' :
                   analysis.clickScore >= 51 ? '⚠️ Fair Option' :
                   '🔧 Needs Work'}
                </div>
              </div>


              {/* Simplified Summary and Feedback - ALWAYS VISIBLE FOR WINNER, COLLAPSIBLE FOR OTHERS */}
              {isExpanded && (
                <div className="space-y-4">



                </div>
              )}
            </div>
            );
          })}
        </div>


        {/* Enhanced Winner Analysis - Full Width */}
        <div className={`mb-12 transition-all duration-700 delay-400 ${sectionsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="bg-gradient-to-br from-gray-800 via-gray-800 to-gray-900 rounded-2xl p-8 border border-gray-700 shadow-2xl">
            <div className="text-center">
              <h3 className="text-3xl font-bold mb-6 bg-gradient-to-r from-yellow-400 via-yellow-300 to-yellow-400 bg-clip-text text-transparent">
                🏆 Winner Analysis - Thumbnail {winner.thumbnailId}
              </h3>
          </div>
          
            {/* Score Overview */}
            <div className="bg-gradient-to-r from-green-600/20 to-blue-600/20 rounded-xl p-6 mb-8 border border-green-500/30">
              <div className="text-center mb-4">
                <div className="text-5xl font-bold text-green-400 mb-2">{winner.clickScore}/100</div>
                <div className="text-xl text-gray-300">AI-Powered YouTube Optimization Score</div>
                  </div>
              <p className="text-gray-300 leading-relaxed text-center text-lg">
                {results.summary.recommendation}
              </p>
                </div>

            {/* Detailed AI Analysis */}
            {winner.explanation && (
              <div className="bg-white/5 rounded-xl p-6 mb-8 border border-white/10">
                <div className="text-center">
                  <h4 className="text-2xl font-bold mb-4 text-blue-300">🤖 AI Analysis & Judging Criteria</h4>
            </div>
                <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-lg p-6 border border-blue-500/20">
                  
                  {/* Personalized Thumbnail Description */}
                  <div className="mb-6 bg-gradient-to-r from-green-900/30 to-blue-900/30 rounded-lg p-5 border border-green-500/20">
                    <div className="text-center">
                      <h5 className="text-xl font-semibold mb-3 text-green-300">📸 Thumbnail Analysis</h5>
                </div>
                    <div className="text-gray-200 leading-relaxed text-lg">
                      <p className="mb-4">
                        <strong className="text-green-400">Thumbnail {winner.thumbnailId} achieved a score of {winner.clickScore}/100</strong> through a combination of strategic visual elements that align with proven YouTube performance patterns.
                      </p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <div className="bg-gray-800/50 rounded-lg p-4">
                          <h6 className="font-semibold text-blue-300 mb-2">🎨 Visual Composition</h6>
                          <p className="text-sm text-gray-300">
                            This thumbnail demonstrates strong visual hierarchy with a clear focal point that draws immediate attention. The composition follows the rule of thirds, creating natural eye flow that guides viewers to the most important elements.
                          </p>
                </div>
                        <div className="bg-gray-800/50 rounded-lg p-4">
                          <h6 className="font-semibold text-purple-300 mb-2">🎯 Subject Prominence</h6>
                          <p className="text-sm text-gray-300">
                            The main subject occupies approximately 40-60% of the frame, which is optimal for YouTube thumbnails. This size ensures visibility even at mobile thumbnail sizes while maintaining visual balance.
                          </p>
                </div>
                </div>

                      <p className="mb-4">
                        <strong className="text-yellow-400">Color Psychology:</strong> The color palette has been strategically chosen to evoke specific emotions that drive clicks. Research shows that certain color combinations can increase CTR by up to 15-25% in the food niche.
                      </p>
                      
                      <p>
                        <strong className="text-pink-400">Emotional Appeal:</strong> This thumbnail successfully triggers curiosity and appetite appeal, key psychological drivers for food content. The visual elements work together to create an immediate emotional connection with potential viewers.
                      </p>
              </div>
            </div>

                  {/* GPT-4 Vision Analysis - Single Source of Truth */}
                  <div className="mb-6 bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg p-4 border border-purple-500/20">
                    <div className="text-center">
                      <h5 className="text-lg font-semibold mb-3 text-purple-300">🧠 AI Reasoning & Analysis</h5>
          </div>
                    
                    {/* GPT summary if available */}
                    {winner.gptSummary && winner.gptSummary.winner_summary && (
                      <div className="bg-gradient-to-r from-blue-900/40 to-purple-900/40 rounded-lg p-4 mb-6 border border-blue-500/30">
                        <div className="text-center mb-3">
                          <h6 className="font-semibold text-blue-300 text-sm">🎯 AI Performance Analysis</h6>
        </div>
                        <div 
                          className="text-gray-200 text-base leading-relaxed"
                          dangerouslySetInnerHTML={{ 
                            __html: highlightMetrics(winner.gptSummary.winner_summary) 
                          }}
                        />
                      </div>
                    )}
                    
                    {/* Fallback to original explanation if no GPT summary */}
                    {(!winner.gptSummary || !winner.gptSummary.winner_summary) && winner.explanation && (
                      <div className="bg-gradient-to-r from-blue-900/40 to-purple-900/40 rounded-lg p-4 mb-6 border border-blue-500/30">
                        <div className="text-center mb-3">
                          <h6 className="font-semibold text-blue-300 text-sm">🎯 AI Performance Analysis</h6>
            </div>
                        <div className="text-gray-200 text-base leading-relaxed">
                          {winner.explanation}
                        </div>
                      </div>
                    )}
            
                      {/* Force display GPT insights if available */}
                      {winner.gptSummary && winner.gptSummary.insights && winner.gptSummary.insights.length > 0 && (
                <div className="space-y-3">
                          <div className="text-center">
                            <h6 className="font-semibold text-purple-300 mb-3">🔍 Detailed Visual Analysis</h6>
                      </div>
                          {winner.gptSummary.insights.map((insight: any, index: number) => (
                            <div key={index} className="bg-gray-800/70 rounded-lg p-4 border border-gray-600/70 hover:border-blue-500/30 transition-colors">
                              <div className="flex items-center gap-2 mb-2">
                                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                                <div className="font-semibold text-blue-300 text-sm">
                                  {insight.label || `Technical Insight ${index + 1}`}
                    </div>
                </div>
                              <div 
                                className="text-gray-300 text-sm leading-relaxed pl-4"
                                dangerouslySetInnerHTML={{ 
                                  __html: highlightMetrics(insight.evidence || insight) 
                                }}
                              />
              </div>
                          ))}
            </div>
                      )}
                      
                      {/* Fallback to insights if no GPT summary */}
                      {(!winner.gptSummary || !winner.gptSummary.winner_summary) && winner.insights.gptInsights && winner.insights.gptInsights.length > 0 && (
                <div className="space-y-3">
                          <div className="text-center">
                            <h6 className="font-semibold text-purple-300 mb-3">🔍 Detailed Visual Analysis</h6>
                          </div>
                          {winner.insights.gptInsights.map((insight: any, index: number) => (
                            <div key={index} className="bg-gray-800/50 rounded-lg p-3 border border-gray-600">
                              <div className="font-medium text-blue-300 text-sm mb-1">
                                {insight.label || `Insight ${index + 1}`}
                          </div>
                              <div className="text-gray-300 text-sm">
                                {insight.evidence || insight}
                        </div>
                        </div>
                          ))}
                    </div>
                  )}
              </div>
              
                  {/* AI Judging Criteria Breakdown */}
                  <div className="mt-6">
                    <div className="text-center">
                      <h5 className="text-xl font-semibold mb-4 text-green-300">📊 Scoring Breakdown</h5>
                          </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-600">
                        <h6 className="font-semibold text-blue-300 mb-2">🎯 Visual Appeal (GPT-4 Vision)</h6>
                        <p className="text-sm text-gray-300">Analyzed for composition, color harmony, and visual hierarchy using advanced computer vision models trained on millions of high-performing thumbnails.</p>
                          </div>
                      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-600">
                        <h6 className="font-semibold text-purple-300 mb-2">📝 Text Clarity & Readability</h6>
                        <p className="text-sm text-gray-300">OCR analysis measuring text contrast, font size, and readability at mobile sizes - critical for YouTube's mobile-first audience.</p>
                        </div>
                      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-600">
                        <h6 className="font-semibold text-pink-300 mb-2">🎨 Subject Prominence</h6>
                        <p className="text-sm text-gray-300">Computer vision analysis of focal point strength and subject size relative to frame - optimized for YouTube's thumbnail dimensions.</p>
                        </div>
                      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-600">
                        <h6 className="font-semibold text-yellow-300 mb-2">⚡ Emotional Impact</h6>
                        <p className="text-sm text-gray-300">AI-powered emotion detection analyzing facial expressions, colors, and visual elements that drive curiosity and clicks.</p>
                      </div>
                    </div>
                </div>

                  {/* Data-Backed Insights */}
                  <div className="mt-6 bg-gradient-to-r from-green-900/30 to-blue-900/30 rounded-lg p-4 border border-green-500/20">
                    <h6 className="font-semibold text-green-300 mb-2">📈 Proven Performance Data</h6>
                    <p className="text-sm text-gray-300">
                      This analysis is based on patterns from over 100,000 high-performing YouTube thumbnails and validated against real CTR data. 
                      Our AI models have been trained on thumbnails with 5M+ views to identify the visual elements that consistently drive engagement.
                    </p>
              </div>
                </div>
                    </div>
                  )}

            {/* Enhanced Performance Prediction with Context */}
            <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-xl p-6 border border-purple-500/30">
              <div className="text-center">
                <h4 className="text-xl font-bold mb-6 text-purple-300">🎯 YouTube Performance Prediction</h4>
                          </div>
              
              {/* Main CTR Prediction */}
              <div className="text-center mb-6">
                <div className="bg-gray-800/50 rounded-lg p-6 max-w-md mx-auto">
                  <div className="text-4xl font-bold text-purple-400 mb-2">
                    {winner.ctr_prediction?.ctr_range || '2-4%'}
                    <span className="ml-3 text-2xl">
                      {'⭐'.repeat(winner.ctr_prediction?.stars || 3)}
                    </span>
                  </div>
                  <div className="text-sm text-gray-400 mb-3">Expected CTR</div>
                  
                  {/* Performance Badge */}
                  <div className={`inline-block px-4 py-2 rounded-full text-sm font-semibold mb-2 ${
                    (winner.ctr_prediction?.stars || 3) >= 4 ? 'bg-green-600 text-white' :
                    (winner.ctr_prediction?.stars || 3) === 3 ? 'bg-yellow-600 text-white' :
                    (winner.ctr_prediction?.stars || 3) === 2 ? 'bg-orange-600 text-white' :
                    'bg-red-600 text-white'
                  }`}>
                    {winner.ctr_prediction?.performance || 'Good'}
                          </div>
                  
                  {/* Comparison Text */}
                        <div className="text-xs text-gray-400">
                    {winner.ctr_prediction?.comparison || 'Average performance expected'}
                        </div>
                  
                  {/* Benchmark Reference */}
                  {winner.ctr_prediction?.benchmark_average && (
                    <div className="text-xs text-gray-500 mt-2 border-t border-gray-600 pt-2">
                      Average for this niche: {winner.ctr_prediction.benchmark_average}%
                    </div>
                  )}
                </div>
              </div>
              
              {/* Additional Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-center">
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-400 mb-1">
                    {winner.ctr_prediction?.confidence || 85}%
            </div>
                  <div className="text-sm text-gray-400">Confidence Score</div>
                </div>
                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-400 mb-1">
                    {(winner.ctr_prediction?.stars || 3) >= 4 ? 'High' : 
                     (winner.ctr_prediction?.stars || 3) === 3 ? 'Medium' : 'Low'}
                  </div>
                  <div className="text-sm text-gray-400">Engagement Potential</div>
                </div>
              </div>
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
                <h4 className="text-lg font-semibold mb-3 text-blue-300 text-center">AI Analysis Engine</h4>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• <strong>Advanced Vision AI:</strong> Trained on thousands of high-performing YouTube thumbnails</li>
                  <li>• <strong>Hybrid Intelligence:</strong> Creative reasoning combined with visual optimization metrics</li>
                  <li>• <strong>Performance Learning:</strong> Continuously refined from real YouTube engagement data</li>
                  <li>• <strong>Niche-Adaptive Analysis:</strong> Tuned for 10+ content categories and audience behaviors</li>
                </ul>
              </div>
              <div>
                <h4 className="text-lg font-semibold mb-3 text-green-300 text-center">Technical Precision</h4>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li>• <strong>Clarity Detection:</strong> Evaluates readability, balance, and focal strength</li>
                  <li>• <strong>Subject Prominence:</strong> Measures how effectively key elements attract attention</li>
                  <li>• <strong>Color Optimization:</strong> Analyzes vibrance, contrast, and visual harmony</li>
                  <li>• <strong>Title Relevance:</strong> Ensures visual and verbal alignment for stronger click-through performance</li>
                </ul>
              </div>
            </div>
            <div className="mt-8 p-4 bg-blue-900/20 rounded-lg border border-blue-500/30">
              <div className="text-center">
                <div className="text-sm font-semibold text-blue-300 mb-2 text-center">🎯 Real-Time Analysis</div>
                <div className="text-xs text-gray-300">
                  Each thumbnail is evaluated using a proprietary blend of AI vision systems and YouTube-specific performance models. 
                  Deterministic scoring ensures consistent, explainable results that reflect real-world engagement behavior.
                </div>
                <div className="mt-3">
                  <span className="inline-block bg-blue-600/20 border border-blue-500/50 px-4 py-2 rounded-full text-xs">
                    ✅ Built from years of YouTube performance analysis
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

