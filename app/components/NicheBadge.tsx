import React from 'react';

interface NicheBadgeProps {
  niche: string;
  onChangeNiche?: () => void;
}

export function NicheBadge({ niche, onChangeNiche }: NicheBadgeProps) {
  const nicheInfo = {
    'gaming': { emoji: '🎮', name: 'Gaming', count: '8,500' },
    'business': { emoji: '💼', name: 'Business & Finance', count: '5,200' },
    'education': { emoji: '🎓', name: 'Education & How-To', count: '6,800' },
    'tech': { emoji: '💻', name: 'Tech & Reviews', count: '4,900' },
    'food': { emoji: '🍳', name: 'Food & Cooking', count: '3,700' },
    'fitness': { emoji: '💪', name: 'Fitness & Health', count: '4,100' },
    'entertainment': { emoji: '🎬', name: 'Entertainment & Vlogs', count: '7,300' },
    'travel': { emoji: '✈️', name: 'Travel & Lifestyle', count: '6,100' },
    'music': { emoji: '🎵', name: 'Music', count: '5,600' },
    'general': { emoji: '📺', name: 'General', count: '12,000' }
  };

  const info = nicheInfo[niche as keyof typeof nicheInfo] || nicheInfo['general'];

  return (
    <div className="bg-gradient-to-r from-blue-900/40 to-purple-900/40 border-2 border-blue-500/50 rounded-xl p-5 mb-6 backdrop-blur-sm">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-2">Analyzed as:</p>
          <div className="flex items-center gap-3">
            <span className="text-3xl">{info.emoji}</span>
            <div>
              <h3 className="text-2xl font-bold text-white">{info.name}</h3>
              <p className="text-sm text-blue-300">
                Compared to {info.count}+ {info.name.toLowerCase()} thumbnails
              </p>
            </div>
          </div>
        </div>
        {onChangeNiche && (
          <button
            onClick={onChangeNiche}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-semibold transition-colors"
          >
            Change Niche
          </button>
        )}
      </div>
    </div>
  );
}