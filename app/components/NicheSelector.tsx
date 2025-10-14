import React, { useState, useEffect } from 'react';

interface Niche {
  id: string;
  name: string;
  emoji: string;
  description: string;
}

interface NicheSelectorProps {
  value?: string;
  onChange?: (nicheId: string) => void;
}

export function NicheSelector({ value, onChange }: NicheSelectorProps) {
  const [niches] = useState<Niche[]>([
    { id: 'gaming', name: 'Gaming', emoji: '🎮', description: 'Video games, streaming, esports' },
    { id: 'business', name: 'Business & Finance', emoji: '💼', description: 'Entrepreneurship, investing, career' },
    { id: 'education', name: 'Education & How-To', emoji: '🎓', description: 'Tutorials, courses, educational content' },
    { id: 'tech', name: 'Tech & Reviews', emoji: '💻', description: 'Technology reviews, gadgets, software' },
    { id: 'food', name: 'Food & Cooking', emoji: '🍳', description: 'Recipes, cooking tutorials, food reviews' },
    { id: 'fitness', name: 'Fitness & Health', emoji: '💪', description: 'Workouts, fitness tips, nutrition' },
    { id: 'entertainment', name: 'Entertainment & Vlogs', emoji: '🎬', description: 'Vlogs, entertainment, comedy' },
    { id: 'travel', name: 'Travel & Lifestyle', emoji: '✈️', description: 'Travel vlogs, destination guides, adventure' },
    { id: 'music', name: 'Music', emoji: '🎵', description: 'Music videos, covers, tutorials' },
    { id: 'general', name: 'General / Other', emoji: '📺', description: 'General content or mixed categories' }
  ]);

  const [selectedNiche, setSelectedNiche] = useState(value || 'general');
  const [showTooltip, setShowTooltip] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('thumbscore_niche');
    if (saved && !value) {
      setSelectedNiche(saved);
      onChange?.(saved);
    }
  }, [value, onChange]);

  const handleChange = (nicheId: string) => {
    setSelectedNiche(nicheId);
    localStorage.setItem('thumbscore_niche', nicheId);
    onChange?.(nicheId);
  };

  const selectedNicheData = niches.find(n => n.id === selectedNiche);

  return (
    <div className="mb-6">
      <label className="text-lg font-semibold text-white mb-2 block flex items-center gap-2">
        <span>📁</span>
        Content Type
      </label>
      
      <div className="relative">
        <select
          value={selectedNiche}
          onChange={(e) => handleChange(e.target.value)}
          className="w-full p-4 pr-10 rounded-xl bg-slate-700 border-2 border-slate-600 text-white text-lg font-medium appearance-none cursor-pointer hover:border-blue-500 transition-colors focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/50"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          {niches.map(niche => (
            <option key={niche.id} value={niche.id}>
              {niche.emoji} {niche.name}
            </option>
          ))}
        </select>
        
        {/* Custom dropdown arrow */}
        <div className="absolute right-4 top-1/2 transform -translate-y-1/2 pointer-events-none">
          <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Description text */}
      {selectedNicheData && (
        <p className="text-sm text-gray-400 mt-2 flex items-start gap-2">
          <span className="text-blue-400">ℹ️</span>
          <span>{selectedNicheData.description} - Optimized scoring for this category</span>
        </p>
      )}

      {/* Info box */}
      <div className="mt-3 bg-blue-900/20 border border-blue-500/30 rounded-lg p-3">
        <p className="text-sm text-blue-200">
          <strong>Why this matters:</strong> Different content types have different thumbnail best practices. 
          Gaming thumbnails need bold emotions, while business content needs professional clarity.
        </p>
      </div>
    </div>
  );
}