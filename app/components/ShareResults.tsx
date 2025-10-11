/**
 * Share Results Component
 * Allows users to share their analysis results
 */

'use client';

import { useState } from 'react';

interface ShareResultsProps {
  sessionId: string;
  winnerScore: number;
  improvement: string;
}

export default function ShareResults({ sessionId, winnerScore, improvement }: ShareResultsProps) {
  const [copied, setCopied] = useState(false);

  const shareUrl = typeof window !== 'undefined' 
    ? `${window.location.origin}/results?id=${sessionId}`
    : '';

  const shareText = `I just analyzed my YouTube thumbnails with Thumbnail Lab! 🎯\n\nWinner scored ${winnerScore}% CTR prediction.\n${improvement}\n\nTry it free:`;

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const shareToTwitter = () => {
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`;
    window.open(twitterUrl, '_blank', 'width=550,height=420');
  };

  const shareToLinkedIn = () => {
    const linkedInUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
    window.open(linkedInUrl, '_blank', 'width=550,height=420');
  };

  const shareToFacebook = () => {
    const facebookUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`;
    window.open(facebookUrl, '_blank', 'width=550,height=420');
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h3 className="text-lg font-bold text-white mb-4">🔗 Share Your Results</h3>
      
      <div className="space-y-4">
        {/* Copy Link */}
        <div className="flex gap-2">
          <input
            type="text"
            value={shareUrl}
            readOnly
            className="flex-1 px-4 py-2 bg-gray-700 text-gray-300 rounded-lg border border-gray-600 text-sm"
          />
          <button
            onClick={copyLink}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-all"
          >
            {copied ? '✓ Copied!' : 'Copy Link'}
          </button>
        </div>

        {/* Social Share Buttons */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <button
            onClick={shareToTwitter}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
          >
            <span className="text-lg">𝕏</span>
            <span className="text-sm font-semibold">Twitter</span>
          </button>
          
          <button
            onClick={shareToLinkedIn}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-700 hover:bg-blue-800 text-white rounded-lg transition-colors"
          >
            <span className="text-lg">in</span>
            <span className="text-sm font-semibold">LinkedIn</span>
          </button>
          
          <button
            onClick={shareToFacebook}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <span className="text-lg">f</span>
            <span className="text-sm font-semibold">Facebook</span>
          </button>
        </div>

        <p className="text-xs text-gray-400 text-center">
          Results are saved for 7 days. Anyone with the link can view this analysis.
        </p>
      </div>
    </div>
  );
}
