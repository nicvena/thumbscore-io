'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { WaitlistModal } from './WaitlistModal'

export function UpgradePrompt() {
  const [showWaitlist, setShowWaitlist] = useState(false)
  const [waitlistCount, setWaitlistCount] = useState(0)
  
  useEffect(() => {
    fetch('/api/waitlist')
      .then(r => r.json())
      .then(data => setWaitlistCount(data.count))
      .catch(() => setWaitlistCount(47))
  }, [])

  return (
    <>
      <div className="p-6 bg-gradient-to-br from-purple-900/50 to-blue-900/50 border border-purple-600/20 rounded-2xl text-center">
        <h2 className="text-2xl font-semibold text-white mb-2">
          Ready for unlimited AI insights? 🚀
        </h2>
        <p className="text-gray-300 mb-4">
          You've experienced the power of Thumbscore. Join the waitlist for early access.
        </p>
        
        {/* Waitlist Count */}
        <div className="mb-6">
          <div className="inline-flex items-center px-3 py-1 bg-purple-600/20 border border-purple-500/30 rounded-full text-sm text-purple-300">
            🎯 {waitlistCount}+ creators waiting
          </div>
        </div>
        
        <div className="space-y-3">
          <button
            onClick={() => setShowWaitlist(true)}
            className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition"
          >
            Join Creator Waitlist ($19/mo)
          </button>
          
          <p className="text-xs text-gray-400">
            Launching in 2-3 weeks • 25 analyses/month
          </p>
          
          <Link
            href="/pricing"
            className="block text-sm text-gray-400 hover:text-gray-300 transition"
          >
            View all plans →
          </Link>
        </div>
      </div>
      
      <WaitlistModal
        isOpen={showWaitlist}
        onClose={() => setShowWaitlist(false)}
        plan="creator"
      />
    </>
  )
}
