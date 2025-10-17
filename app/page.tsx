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
    <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25]">
      <div className="container mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center mb-20">
          {/* Logo & Title */}
          <h1 className="text-6xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-[#6a5af9] via-[#1de9b6] to-[#6a5af9] bg-clip-text text-transparent">
            Thumbscore.io
          </h1>
          <p className="text-2xl font-semibold text-white mb-8">
            YouTube Click Optimization
          </p>
          
          {/* Feature Tags */}
          <div className="flex justify-center mb-8">
            <div className="inline-flex bg-blue-600/20 border border-blue-500/50 rounded-full px-6 py-3">
              <span className="text-blue-300 font-semibold text-lg">AI-powered • Instant Results • Niche-Specific</span>
            </div>
          </div>
          
          {/* Main Headline */}
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 leading-tight">
            Optimize Your YouTube Thumbnails for Maximum Clicks
          </h2>
          
          {/* Description */}
          <p className="text-xl text-cyan-300 font-semibold mb-6 max-w-3xl mx-auto">
            Upload 3 thumbnail options. Our AI analyzes them specifically for YouTube click-through rates. Get results in 10 seconds.
          </p>
          
          {/* Pricing Info */}
          <p className="text-lg text-gray-300 mb-12">
            No credit card required • 5 free analyses per month
          </p>
          
          {/* CTA Buttons */}
          <div className="flex justify-center gap-6 flex-wrap mb-6">
            <a
              href="/upload"
              onClick={handleUploadClick}
              className="px-10 py-5 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white rounded-xl hover:shadow-xl hover:shadow-cyan-500/50 transition-all text-xl font-semibold transform hover:scale-105"
            >
              Analyze 3 Thumbnails Free →
          </a>
          <a
              href="/pricing"
              onClick={handlePricingClick}
              className="px-8 py-5 bg-white/10 text-white rounded-xl hover:bg-white/20 transition-colors text-lg font-semibold backdrop-blur-sm border border-white/20"
            >
              Pricing
        </a>
        <a
              href="/faq"
              onClick={handleFaqClick}
              className="px-8 py-5 bg-white/10 text-white rounded-xl hover:bg-white/20 transition-colors text-lg font-semibold backdrop-blur-sm border border-white/20"
            >
              FAQ
            </a>
          </div>
          
          {/* Bottom CTA Text */}
          <p className="text-sm text-gray-400">
            No credit card required • Results in 10 seconds
          </p>
        </div>

        {/* Key Benefits */}
        <div className="mb-20">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-cyan-500/50 transition-colors text-center group">
              <div className="text-5xl mb-6 group-hover:scale-110 transition-transform">⚡</div>
              <h4 className="text-xl font-semibold text-white mb-3">10 Second Results</h4>
              <p className="text-gray-300">Instant analysis, no waiting</p>
            </div>
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-cyan-500/50 transition-colors text-center group">
              <div className="text-5xl mb-6 group-hover:scale-110 transition-transform">🎯</div>
              <h4 className="text-xl font-semibold text-white mb-3">Niche-Specific</h4>
              <p className="text-gray-300">Optimized for your content type</p>
            </div>
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-cyan-500/50 transition-colors text-center group">
              <div className="text-5xl mb-6 group-hover:scale-110 transition-transform">📊</div>
              <h4 className="text-xl font-semibold text-white mb-3">Consistent Scores</h4>
              <p className="text-gray-300">Same thumbnail = same score</p>
            </div>
          </div>
        </div>

        {/* Detailed How ThumbScore Works */}
        <div className="mb-20">
          <h3 className="text-4xl font-bold text-white mb-16 text-center">
            How ThumbScore Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-cyan-500/50 transition-colors group">
              <div className="text-6xl mb-6 font-bold text-cyan-400 group-hover:scale-110 transition-transform">1</div>
              <h4 className="text-2xl font-semibold mb-4 text-white">Upload 3 Thumbnails</h4>
              <p className="text-gray-300 text-lg">Drag & drop or click to upload your thumbnail options. Supports JPG, PNG.</p>
            </div>
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-cyan-500/50 transition-colors group">
              <div className="text-6xl mb-6 font-bold text-cyan-400 group-hover:scale-110 transition-transform">2</div>
              <h4 className="text-2xl font-semibold mb-4 text-white">Select Your Niche</h4>
              <p className="text-gray-300 text-lg">Tell us your content category. Our AI adjusts its analysis to match what works in your niche.</p>
            </div>
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-cyan-500/50 transition-colors group">
              <div className="text-6xl mb-6 font-bold text-cyan-400 group-hover:scale-110 transition-transform">3</div>
              <h4 className="text-2xl font-semibold mb-4 text-white">Get Instant Analysis</h4>
              <p className="text-gray-300 text-lg">In 10 seconds, see which thumbnail wins with scores, explanations, and actionable suggestions.</p>
            </div>
          </div>
        </div>


        {/* Niche-Specific Intelligence */}
        <div className="mb-20">
          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 max-w-5xl mx-auto">
            <h3 className="text-3xl font-bold text-white mb-6 text-center">
              Niche-Specific Intelligence
            </h3>
            <p className="text-lg text-gray-300 mb-8 text-center max-w-4xl mx-auto">
              Unlike generic thumbnail analyzers, ThumbScore uses specialized analysis trained across 10 YouTube content categories. What works for gaming doesn't work for education.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">🎮</div>
                <div className="text-lg font-semibold text-white">Gaming</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">💼</div>
                <div className="text-lg font-semibold text-white">Business</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">🎓</div>
                <div className="text-lg font-semibold text-white">Education</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">💻</div>
                <div className="text-lg font-semibold text-white">Tech</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">🍳</div>
                <div className="text-lg font-semibold text-white">Food</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">💪</div>
                <div className="text-lg font-semibold text-white">Fitness</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">🎬</div>
                <div className="text-lg font-semibold text-white">Entertainment</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">✈️</div>
                <div className="text-lg font-semibold text-white">Travel</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">🎵</div>
                <div className="text-lg font-semibold text-white">Music</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center hover:bg-white/20 transition-colors border border-white/20 group">
                <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">📺</div>
                <div className="text-lg font-semibold text-white">General</div>
              </div>
            </div>
          </div>
        </div>

        {/* Final CTA Section */}
        <div className="mb-16">
          <div className="bg-gradient-to-r from-[#6a5af9]/20 to-[#1de9b6]/20 backdrop-blur-sm rounded-xl p-12 border border-white/10 max-w-5xl mx-auto text-center">
            <h3 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Stop Losing Views to Bad Thumbnails
            </h3>
            <p className="text-xl text-gray-300 mb-10 max-w-3xl mx-auto">
              Join thousands of creators who've already maximized their CTR
            </p>
            <a
              href="/upload"
              onClick={handleUploadClick}
              className="inline-block px-12 py-6 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white rounded-xl hover:shadow-xl hover:shadow-cyan-500/50 transition-all text-2xl font-semibold transform hover:scale-105"
            >
              Start Analyzing Free →
            </a>
            <p className="text-sm text-gray-400 mt-6">
              No credit card required • Results in 10 seconds
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-gray-500 pb-8">
          <p>© 2025 Thumbscore.io — AI-powered thumbnail scoring engine</p>
        </div>
    </div>
    </main>
  );
}
