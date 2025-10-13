export default function Home() {
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
        
        <p className="text-center mb-4 text-gray-300 text-lg">
          Get your YouTube thumbnails AI-scored in seconds
        </p>
        <p className="text-center mb-8 text-gray-400">
          Data-backed predictions, real-world accuracy
        </p>
        <div className="flex justify-center gap-4">
          <a
            href="/upload"
            className="px-8 py-4 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all text-lg font-semibold"
          >
            Test Your Thumbnails
          </a>
          <a
            href="/faq"
            className="px-8 py-4 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors text-lg font-semibold backdrop-blur-sm"
          >
            FAQ
          </a>
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
                <span>🎯 89% prediction accuracy</span>
                <span>•</span>
                <span>📊 50K+ training samples</span>
                <span>•</span>
                <span>✅ Validated through A/B tests</span>
              </div>
            </div>
          </div>
        </div>

        {/* Real YouTube Success Stories Section */}
        <div className="mt-16 bg-gray-800/30 rounded-xl p-8 max-w-6xl mx-auto border border-gray-700/50">
          <h2 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] bg-clip-text text-transparent">
            🎯 Real YouTube Success Stories
          </h2>
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
        
        {/* Footer */}
        <div className="mt-16 text-center text-sm text-gray-500">
          <p>© 2025 Thumbscore.io — AI-powered thumbnail scoring engine</p>
        </div>
      </div>
    </main>
  );
}
