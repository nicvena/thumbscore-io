'use client';

import { useEffect } from 'react';
import { useAnalytics } from '@/lib/hooks/useAnalytics';

export default function Home() {
  const analytics = useAnalytics();

  useEffect(() => {
    // Track homepage visit
    analytics.trackEvent('homepage_visit', {
      event_category: 'engagement',
      event_label: 'homepage',
    });
  }, [analytics]);

  const handleUploadClick = () => {
    analytics.trackEvent('cta_click', {
      event_category: 'conversion',
      event_label: 'upload_button',
      custom_parameter_1: 'homepage',
    });
  };

  const handleFaqClick = () => {
    analytics.trackEvent('cta_click', {
      event_category: 'engagement',
      event_label: 'faq_button',
      custom_parameter_1: 'homepage',
    });
  };

  const handlePricingClick = () => {
    analytics.trackEvent('cta_click', {
      event_category: 'engagement',
      event_label: 'pricing_button',
      custom_parameter_1: 'homepage',
    });
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex flex-col items-center justify-center p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm">
        {/* Logo & Title */}
        <div className="text-center mb-8">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-[#6a5af9] via-[#1de9b6] to-[#6a5af9] bg-clip-text text-transparent">
            Thumbscore.io
          </h1>
          <p className="text-xl font-semibold text-cyan-400">
            AI Thumbnail Scoring
          </p>
        </div>
        
        <div className="text-center mb-4">
          <div className="inline-block bg-blue-600/20 border border-blue-500 rounded-full px-4 py-2 mb-4">
            <span className="text-blue-300 font-semibold">🧪 BETA</span>
            <span className="text-gray-300 ml-2">AI-powered thumbnail analysis trained on real YouTube data</span>
          </div>
          <h2 className="text-3xl font-bold text-white mb-4">
            Stop Guessing Which Thumbnail Will Get Clicks
          </h2>
          <p className="text-xl text-gray-300 mb-4">
            You spent hours editing your video. Don't let a bad thumbnail kill your views. 82% of viewers decide whether to click based on your thumbnail alone.
          </p>
          <p className="text-lg text-cyan-300 font-semibold">
            Upload 3 options. Our AI tells you which one will get the most clicks. In under 10 seconds.
          </p>
        </div>
        <p className="text-center mb-8 text-gray-400">
          Data-backed predictions, continuously improving
        </p>
        <div className="flex justify-center gap-4 flex-wrap">
          <a
            href="/upload"
            onClick={handleUploadClick}
            className="px-8 py-4 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all text-lg font-semibold"
          >
            Find Your Winning Thumbnail Free →
          </a>
          <a
            href="/pricing"
            onClick={handlePricingClick}
            className="px-8 py-4 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors text-lg font-semibold backdrop-blur-sm"
          >
            Pricing
          </a>
          <a
            href="/faq"
            onClick={handleFaqClick}
            className="px-8 py-4 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors text-lg font-semibold backdrop-blur-sm"
          >
            FAQ
          </a>
        </div>
        <div className="text-center mt-3">
          <p className="text-sm text-gray-400">
            No credit card • 5 free analyses • See results instantly
          </p>
        </div>
        
        {/* Pain Points Section */}
        <div className="mt-16 bg-red-900/20 backdrop-blur-sm rounded-lg p-8 border border-red-500/30 max-w-4xl mx-auto">
          <h3 className="text-2xl font-bold text-white mb-8 text-center">
            Why Most YouTubers Struggle With Thumbnails
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div className="bg-red-800/20 rounded-lg p-6 border border-red-500/20">
              <div className="text-4xl mb-4">😫</div>
              <h4 className="text-xl font-semibold mb-2 text-red-300">You're Guessing</h4>
              <p className="text-gray-300">Going with gut feeling costs you thousands of potential views. What looks good to you might bomb with your audience.</p>
            </div>
            <div className="bg-red-800/20 rounded-lg p-6 border border-red-500/20">
              <div className="text-4xl mb-4">⏰</div>
              <h4 className="text-xl font-semibold mb-2 text-red-300">A/B Testing Takes Forever</h4>
              <p className="text-gray-300">Traditional thumbnail testing means waiting days or weeks for meaningful data. Your momentum dies while you wait.</p>
            </div>
            <div className="bg-red-800/20 rounded-lg p-6 border border-red-500/20">
              <div className="text-4xl mb-4">📉</div>
              <h4 className="text-xl font-semibold mb-2 text-red-300">Bad Thumbnails Kill Great Videos</h4>
              <p className="text-gray-300">Your best content gets buried because the thumbnail didn't grab attention. All that editing work goes to waste.</p>
            </div>
          </div>
        </div>

        {/* Comparison Section */}
        <div className="mt-16 bg-gradient-to-r from-red-900/10 via-gray-900/20 to-green-900/10 backdrop-blur-sm rounded-lg p-8 border border-white/10 max-w-5xl mx-auto">
          <h3 className="text-2xl font-bold text-white mb-8 text-center">
            Old Way vs ThumbScore Way
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-red-900/20 rounded-lg p-6 border border-red-500/30">
              <h4 className="text-xl font-semibold mb-4 text-red-300 text-center">😤 The Old Way</h4>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-red-400">❌</span>
                  <span className="text-gray-300">Upload and pray it works</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-red-400">❌</span>
                  <span className="text-gray-300">Wait weeks for A/B test results</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-red-400">❌</span>
                  <span className="text-gray-300">Miss algorithm window</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-red-400">❌</span>
                  <span className="text-gray-300">Lose thousands of potential views</span>
                </div>
              </div>
            </div>
            <div className="bg-green-900/20 rounded-lg p-6 border border-green-500/30">
              <h4 className="text-xl font-semibold mb-4 text-green-300 text-center">🚀 ThumbScore Way</h4>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-green-400">✅</span>
                  <span className="text-gray-300">Know which thumbnail wins before uploading</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-green-400">✅</span>
                  <span className="text-gray-300">Get instant AI predictions in 10 seconds</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-green-400">✅</span>
                  <span className="text-gray-300">Publish with confidence immediately</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-green-400">✅</span>
                  <span className="text-gray-300">Maximize views from day one</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 hover:border-cyan-500/50 transition-colors">
            <div className="text-3xl mb-4">📸</div>
            <h3 className="text-xl font-semibold mb-2 text-white">Upload 3 Options</h3>
            <p className="text-gray-400">Upload up to 3 different thumbnail variations to compare</p>
          </div>
          <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 hover:border-cyan-500/50 transition-colors">
            <div className="text-3xl mb-4">🤖</div>
            <h3 className="text-xl font-semibold mb-2 text-white">AI Analysis</h3>
            <p className="text-gray-400">Visual quality, power words, and similarity to 2000+ top thumbnails</p>
          </div>
          <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 hover:border-cyan-500/50 transition-colors">
            <div className="text-3xl mb-4">📈</div>
            <h3 className="text-xl font-semibold mb-2 text-white">Get Results</h3>
            <p className="text-gray-400">See which thumbnail is predicted to perform best</p>
          </div>
        </div>

        {/* How ThumbScore Works Section */}
        <div className="mt-20 bg-black/20 backdrop-blur-sm rounded-lg px-8 py-8 max-w-4xl mx-auto border border-white/10">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-8 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] bg-clip-text text-transparent">
              📊 How ThumbScore™ Works
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-gray-300">
                  <span className="text-cyan-400">•</span>
                  <span>Face emotion analysis</span>
                </div>
                <div className="flex items-center gap-2 text-gray-300">
                  <span className="text-cyan-400">•</span>
                  <span>Color contrast optimization</span>
                </div>
                <div className="flex items-center gap-2 text-gray-300">
                  <span className="text-cyan-400">•</span>
                  <span>Text readability scoring</span>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-gray-300">
                  <span className="text-cyan-400">•</span>
                  <span>Visual hierarchy assessment</span>
                </div>
                <div className="flex items-center gap-2 text-gray-300">
                  <span className="text-cyan-400">•</span>
                  <span>Power word detection</span>
                </div>
                <div className="flex items-center gap-2 text-gray-300">
                  <span className="text-cyan-400">•</span>
                  <span>Mobile viewing optimization</span>
                </div>
              </div>
            </div>
            <div className="mt-6 pt-4 border-t border-white/10">
              <div className="flex items-center justify-center gap-4 text-sm text-gray-400">
                <span>📊 High-Accuracy Predictions</span>
                <span>•</span>
                <span>📊 50K+ training samples</span>
                <span>•</span>
                <span>✅ Trained on trending YouTube thumbnails</span>
              </div>
              <p className="text-center text-xs text-gray-500 mt-2">
                Trained on 50K+ thumbnails, continuously improving with user data
              </p>
            </div>
          </div>
        </div>

        {/* Niche-Specific Intelligence Section */}
        <div className="mt-16 bg-slate-800 rounded-xl p-6 max-w-4xl mx-auto">
          <h3 className="text-2xl font-bold text-white mb-4">
            🎯 Niche-Specific Intelligence
          </h3>
          <p className="text-gray-300 mb-4">
            Unlike generic thumbnail analyzers, ThumbScore uses specialized AI models 
            trained across 10 YouTube content categories. ThumbScore is in active development - our AI learns 
            from thousands of trending thumbnails daily, continuously improving as more creators use the platform.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">🎮</div>
              <div className="text-sm font-semibold text-white">Gaming</div>
              <div className="text-xs text-gray-400">8,500+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">💼</div>
              <div className="text-sm font-semibold text-white">Business</div>
              <div className="text-xs text-gray-400">5,200+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">🎓</div>
              <div className="text-sm font-semibold text-white">Education</div>
              <div className="text-xs text-gray-400">6,800+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">💻</div>
              <div className="text-sm font-semibold text-white">Tech</div>
              <div className="text-xs text-gray-400">4,900+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">🍳</div>
              <div className="text-sm font-semibold text-white">Food</div>
              <div className="text-xs text-gray-400">3,700+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">💪</div>
              <div className="text-sm font-semibold text-white">Fitness</div>
              <div className="text-xs text-gray-400">4,100+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">🎬</div>
              <div className="text-sm font-semibold text-white">Entertainment</div>
              <div className="text-xs text-gray-400">7,300+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">✈️</div>
              <div className="text-sm font-semibold text-white">Travel</div>
              <div className="text-xs text-gray-400">4,500+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">🎵</div>
              <div className="text-sm font-semibold text-white">Music</div>
              <div className="text-xs text-gray-400">5,600+ samples</div>
            </div>
            <div className="bg-slate-700 rounded-lg p-4 text-center hover:bg-slate-600 transition-colors">
              <div className="text-3xl mb-2">📺</div>
              <div className="text-sm font-semibold text-white">General</div>
              <div className="text-xs text-gray-400">12,000+ samples</div>
            </div>
          </div>
          <div className="mt-4 text-center">
            <p className="text-sm text-blue-300">
              Each niche has custom scoring weights, power words, and visual preferences trained across 10 YouTube content categories
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Total: 50,000+ samples across 10 specialized categories
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-16 text-center text-sm text-gray-500">
          <p>© 2025 Thumbscore.io — AI-powered thumbnail scoring engine</p>
        </div>
        </div>
      </main>
  );
}
