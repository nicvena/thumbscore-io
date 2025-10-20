import React from 'react';

interface WinnerCardProps {
  winnerName: string;
  score: number;
  label: "🔥 Strong Performer" | "💡 Solid Choice" | "⚠️ Needs Work";
}

export default function WinnerCard({ winnerName, score, label }: WinnerCardProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-400";
    if (score >= 65) return "text-yellow-400";
    if (score >= 50) return "text-orange-400";
    return "text-red-400";
  };

  const getGlowColor = (score: number) => {
    if (score >= 80) return "shadow-green-500/20";
    if (score >= 65) return "shadow-yellow-500/20";
    if (score >= 50) return "shadow-orange-500/20";
    return "shadow-red-500/20";
  };

  return (
    <div className="relative bg-gradient-to-br from-gray-800/50 to-gray-900/50 rounded-2xl p-8 border border-gray-700/50 overflow-hidden">
      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 rounded-2xl"></div>
      
      {/* Content */}
      <div className="relative z-10 text-center">
        {/* Trophy icon and winner label */}
        <div className="flex items-center justify-center gap-3 mb-6">
          <div className="text-3xl">🏆</div>
          <h1 className="text-2xl font-bold text-white">
            Winner: {winnerName}
          </h1>
        </div>

        {/* Big score */}
        <div className={`text-7xl md:text-8xl font-black tracking-tight ${getScoreColor(score)} drop-shadow-lg ${getGlowColor(score)}`}>
          {Math.round(score)}
        </div>
        
        {/* Score label */}
        <div className="mt-4">
          <span className="inline-block px-4 py-2 bg-gray-700/50 rounded-full text-sm font-medium text-gray-300 border border-gray-600/50">
            {label}
          </span>
        </div>

        {/* Score out of 100 */}
        <div className="mt-2 text-sm text-gray-400">
          out of 100
        </div>
      </div>

      {/* Subtle glow ring */}
      <div className={`absolute inset-0 rounded-2xl ring-1 ring-opacity-20 ${getGlowColor(score).replace('shadow-', 'ring-')}`}></div>
    </div>
  );
}
