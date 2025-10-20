'use client'

import Link from 'next/link'

export function UpgradePrompt() {
  return (
    <div className="p-6 bg-gradient-to-br from-purple-900/50 to-blue-900/50 border border-purple-600/20 rounded-2xl text-center">
      <h2 className="text-2xl font-semibold text-white mb-2">
        Ready for unlimited AI insights? 🚀
      </h2>
      <p className="text-gray-300 mb-6">
        You've experienced the power of Thumbscore. Unlock unlimited analyses and advanced features.
      </p>
      <div className="flex justify-center gap-4">
        <Link
          href="/pricing"
          className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition"
        >
          Upgrade for $19/mo →
        </Link>
      </div>
      <p className="text-xs text-gray-500 mt-4">
        Already subscribed? Sign in to access your full plan.
      </p>
    </div>
  )
}
