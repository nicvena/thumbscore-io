import React from 'react';

interface NicheInsightsProps {
  insights: string[];
}

export function NicheInsights({ insights }: NicheInsightsProps) {
  if (!insights || insights.length === 0) return null;

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 mb-6">
      <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <span>💡</span>
        Niche-Specific Insights
      </h3>
      <div className="space-y-3">
        {insights.map((insight, index) => (
          <div
            key={index}
            className="bg-slate-700/50 border-l-4 border-blue-500 rounded-lg p-4"
          >
            <p className="text-gray-200">{insight}</p>
          </div>
        ))}
      </div>
    </div>
  );
}