// Removed unused useState import

interface PowerWord {
  word: string;
  tier: number | string;
  impact: number;
}

interface PowerWordAnalysisData {
  score: number;
  found_words: PowerWord[];
  recommendation: string;
  warnings: string[];
  missing_opportunities: string[];
  breakdown: {
    tier1_count: number;
    tier2_count: number;
    tier3_count: number;
    tier4_count: number;
    niche_count: number;
    negative_count: number;
  };
  caps_percentage: number;
}

interface PowerWordAnalysisProps {
  analysis: PowerWordAnalysisData;
  ocrText: string;
}

export default function PowerWordAnalysis({ analysis, ocrText }: PowerWordAnalysisProps) {
  const { score, found_words, recommendation, warnings } = analysis;

  // Determine score color and gradient
  const getScoreStyle = (score: number) => {
    if (score >= 85) {
      return {
        color: 'text-green-400',
        bgGradient: 'from-green-500/20 to-emerald-500/20',
        borderColor: 'border-green-500/30',
        progressBg: 'bg-gradient-to-r from-green-500 to-emerald-400'
      };
    } else if (score >= 70) {
      return {
        color: 'text-blue-400',
        bgGradient: 'from-blue-500/20 to-cyan-500/20',
        borderColor: 'border-blue-500/30',
        progressBg: 'bg-gradient-to-r from-blue-500 to-cyan-400'
      };
    } else if (score >= 55) {
      return {
        color: 'text-yellow-400',
        bgGradient: 'from-yellow-500/20 to-amber-500/20',
        borderColor: 'border-yellow-500/30',
        progressBg: 'bg-gradient-to-r from-yellow-500 to-amber-400'
      };
    } else {
      return {
        color: 'text-red-400',
        bgGradient: 'from-red-500/20 to-rose-500/20',
        borderColor: 'border-red-500/30',
        progressBg: 'bg-gradient-to-r from-red-500 to-rose-400'
      };
    }
  };

  const scoreStyle = getScoreStyle(score);

  // Get badge style for power word tier
  const getBadgeStyle = (word: PowerWord) => {
    if (word.tier === 1) {
      return {
        bg: 'bg-green-500/20',
        border: 'border-green-500/50',
        text: 'text-green-300',
        icon: '⭐'
      };
    } else if (word.tier === 2) {
      return {
        bg: 'bg-blue-500/20',
        border: 'border-blue-500/50',
        text: 'text-blue-300',
        icon: '💎'
      };
    } else if (word.tier === 3) {
      return {
        bg: 'bg-purple-500/20',
        border: 'border-purple-500/50',
        text: 'text-purple-300',
        icon: '✨'
      };
    } else if (word.tier === 4) {
      return {
        bg: 'bg-orange-500/20',
        border: 'border-orange-500/50',
        text: 'text-orange-300',
        icon: '📊'
      };
    } else if (word.tier === 'niche') {
      return {
        bg: 'bg-cyan-500/20',
        border: 'border-cyan-500/50',
        text: 'text-cyan-300',
        icon: '🎮'
      };
    } else if (word.tier === 'negative') {
      return {
        bg: 'bg-red-500/20',
        border: 'border-red-500/50',
        text: 'text-red-300',
        icon: '⚠️'
      };
    } else {
      return {
        bg: 'bg-gray-500/20',
        border: 'border-gray-500/50',
        text: 'text-gray-300',
        icon: '•'
      };
    }
  };

  // Get recommendation box style
  const getRecommendationStyle = (score: number) => {
    if (score >= 85) {
      return {
        bg: 'bg-green-500/10',
        border: 'border-green-500/30',
        icon: '🔥',
        iconColor: 'text-green-400'
      };
    } else if (score >= 70) {
      return {
        bg: 'bg-blue-500/10',
        border: 'border-blue-500/30',
        icon: '✅',
        iconColor: 'text-blue-400'
      };
    } else {
      return {
        bg: 'bg-yellow-500/10',
        border: 'border-yellow-500/30',
        icon: '⚠️',
        iconColor: 'text-yellow-400'
      };
    }
  };

  const recStyle = getRecommendationStyle(score);

  return (
    <div className="space-y-6">
      {/* Header with Score - COMPACT */}
      <div className={`p-5 rounded-xl border backdrop-blur-sm bg-gradient-to-br ${scoreStyle.bgGradient} ${scoreStyle.borderColor}`}>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xl font-bold text-white">📝 Text Language Analysis</h4>
          <div className="flex items-baseline gap-1">
            <div className={`text-5xl font-bold ${scoreStyle.color}`}>{score}</div>
            <div className="text-sm text-gray-400">/ 100</div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-700/50 rounded-full h-3 mb-3 overflow-hidden">
          <div
            className={`h-full ${scoreStyle.progressBg} transition-all duration-1000 ease-out`}
            style={{ width: `${score}%` }}
          />
        </div>

        {/* OCR Text Display */}
        {ocrText && (
          <div className="p-2 bg-black/30 rounded-lg">
            <div className="text-xs text-gray-500 mb-1">Detected text:</div>
            <div className="text-sm text-gray-300 font-mono">&quot;{ocrText}&quot;</div>
          </div>
        )}
      </div>

      {/* Recommendation Box - COMPACT */}
      <div className={`p-4 rounded-xl border backdrop-blur-sm ${recStyle.bg} ${recStyle.border}`}>
        <div className="flex items-start gap-2">
          <div className={`text-xl ${recStyle.iconColor}`}>{recStyle.icon}</div>
          <div className="flex-1">
            <div className="text-sm text-gray-300">{recommendation}</div>
          </div>
        </div>
      </div>

      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="p-4 rounded-xl border border-orange-500/30 bg-orange-500/10 backdrop-blur-sm">
          <div className="flex items-start gap-3">
            <div className="text-xl text-orange-400">⚠️</div>
            <div className="flex-1">
              <div className="font-semibold text-orange-300 mb-2">Clickbait Warnings</div>
              <ul className="space-y-1">
                {warnings.map((warning, index) => (
                  <li key={index} className="text-sm text-orange-200">
                    • {warning}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Found Power Words - COMPACT (max 8 badges) */}
      {found_words && found_words.length > 0 && (
        <div className="p-4 rounded-xl border border-white/10 bg-white/5 backdrop-blur-sm">
          <h5 className="text-base font-semibold text-white mb-3">
            Found Power Words ({Math.min(found_words.length, 8)}{found_words.length > 8 ? '+' : ''})
          </h5>
          <div className="flex flex-wrap gap-2">
            {found_words.slice(0, 8).map((word, index) => {
              const badgeStyle = getBadgeStyle(word);
              return (
                <div
                  key={index}
                  className={`px-3 py-2 rounded-lg border ${badgeStyle.bg} ${badgeStyle.border} ${badgeStyle.text} text-sm font-medium transition-all hover:scale-105`}
                >
                  <span className="mr-1">{badgeStyle.icon}</span>
                  {word.word}
                  <span className="ml-2 opacity-75">
                    {word.impact > 0 ? `+${word.impact}` : word.impact}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

