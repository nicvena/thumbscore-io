/**
 * Simple Feedback Widget
 * Collects user feedback on analysis quality
 */

'use client';

import { useState } from 'react';

interface FeedbackWidgetProps {
  sessionId: string;
  winnerId: number;
}

export default function FeedbackWidget({ sessionId, winnerId }: FeedbackWidgetProps) {
  const [feedback, setFeedback] = useState<'helpful' | 'not-helpful' | null>(null);
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [showComment, setShowComment] = useState(false);

  const handleFeedback = async (isHelpful: boolean) => {
    const feedbackType = isHelpful ? 'helpful' : 'not-helpful';
    setFeedback(feedbackType);
    setShowComment(true);

    // Send feedback (in production, send to analytics/database)
    console.log('Feedback:', {
      sessionId,
      winnerId,
      helpful: isHelpful,
      timestamp: new Date().toISOString()
    });

    // Simple analytics tracking
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'feedback', {
        session_id: sessionId,
        helpful: isHelpful
      });
    }
  };

  const handleSubmitComment = async () => {
    if (!comment.trim()) return;

    // Send comment (in production, send to backend)
    console.log('Feedback comment:', {
      sessionId,
      winnerId,
      helpful: feedback === 'helpful',
      comment,
      timestamp: new Date().toISOString()
    });

    setSubmitted(true);
    
    // Hide after 3 seconds
    setTimeout(() => {
      setShowComment(false);
      setSubmitted(false);
    }, 3000);
  };

  return (
    <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-6 border border-gray-700">
      <h3 className="text-lg font-bold text-white mb-4 text-center">📝 Was this analysis helpful?</h3>
      
      {!feedback && (
        <div className="flex gap-4">
          <button
            onClick={() => handleFeedback(true)}
            className="flex-1 px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold transition-all transform hover:scale-105"
          >
            👍 Yes, helpful!
          </button>
          <button
            onClick={() => handleFeedback(false)}
            className="flex-1 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-all transform hover:scale-105"
          >
            👎 Not really
          </button>
        </div>
      )}

      {feedback && !submitted && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-white">
            {feedback === 'helpful' ? (
              <span className="text-green-400">✓ Thanks for the feedback!</span>
            ) : (
              <span className="text-yellow-400">⚠️ Sorry it wasn&apos;t helpful</span>
            )}
          </div>

          {showComment && (
            <div className="space-y-3 animate-fade-in">
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder={
                  feedback === 'helpful' 
                    ? "What did you like most? (optional)"
                    : "What could we improve? (optional)"
                }
                className="w-full p-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none resize-none"
                rows={3}
              />
              <div className="flex gap-2">
                <button
                  onClick={handleSubmitComment}
                  disabled={!comment.trim()}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm font-semibold transition-colors"
                >
                  Send Feedback
                </button>
                <button
                  onClick={() => setShowComment(false)}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm transition-colors"
                >
                  Skip
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {submitted && (
        <div className="text-center py-4 animate-fade-in">
          <div className="text-2xl mb-2">🎉</div>
          <p className="text-green-400 font-semibold">Thank you! Your feedback helps us improve.</p>
        </div>
      )}
    </div>
  );
}
