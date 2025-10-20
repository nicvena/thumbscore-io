import React from 'react';

interface Insight {
  label: string;
  evidence: string;
}

interface AnalysisSummaryProps {
  summary: string;
  insights: Insight[];
}

export default function AnalysisSummary({ summary, insights }: AnalysisSummaryProps) {
  return (
    <div className="bg-gray-800/30 rounded-xl p-6 border border-gray-700/50">
      <h2 className="text-xl font-semibold mb-4 text-center text-gray-200">
        Why this wins
      </h2>
      
      {/* Winner explanation */}
      <p className="text-lg leading-7 text-zinc-200 mb-6 text-center">
        {summary}
      </p>

      {/* Insights */}
      {insights && insights.length > 0 ? (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide text-center">
            Key Strengths
          </h3>
          <ul className="space-y-2">
            {insights.slice(0, 3).map((insight, index) => (
              <li key={index} className="flex items-start gap-3 text-gray-300">
                <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                <div>
                  <strong className="text-blue-300">{insight.label}</strong>
                  <span className="text-gray-400"> — </span>
                  <span>{insight.evidence}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide text-center">
            Analysis Complete
          </h3>
          <div className="text-center text-gray-400 text-sm">
            Detailed insights will be available in future updates.
          </div>
        </div>
      )}
    </div>
  );
}
