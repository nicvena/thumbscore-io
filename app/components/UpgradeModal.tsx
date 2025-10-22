'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { WaitlistModal } from './WaitlistModal'

interface UpgradeModalProps {
  isOpen: boolean
  onClose: () => void
}

export function UpgradeModal({ isOpen, onClose }: UpgradeModalProps) {
  const [showWaitlist, setShowWaitlist] = useState(false)
  const [waitlistCount, setWaitlistCount] = useState(0)

  useEffect(() => {
    if (isOpen) {
      fetch('/api/waitlist')
        .then(r => r.json())
        .then(data => setWaitlistCount(data.count))
        .catch(() => setWaitlistCount(47))
    }
  }, [isOpen])

  if (!isOpen) return null

  const handleJoinWaitlist = () => {
    setShowWaitlist(true)
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-8 max-w-md w-full border border-gray-700">
        <div className="text-center">
          {/* Icon */}
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>

          {/* Content */}
          <h2 className="text-2xl font-bold text-white mb-4">
            Ready for unlimited AI insights?
          </h2>
          
          <p className="text-gray-300 mb-4">
            You've used your free analysis. Join the Creator waitlist for early access to unlimited analyses.
          </p>
          
          {/* Waitlist Count */}
          <div className="mb-6">
            <div className="inline-flex items-center px-3 py-1 bg-purple-600/20 border border-purple-500/30 rounded-full text-sm text-purple-300">
              🎯 {waitlistCount}+ creators waiting
            </div>
          </div>

          {/* Features */}
          <div className="text-left mb-6 space-y-2">
            <div className="flex items-center text-sm text-gray-300">
              <span className="text-green-400 mr-2">✓</span>
              25 analyses per month
            </div>
            <div className="flex items-center text-sm text-gray-300">
              <span className="text-green-400 mr-2">✓</span>
              Full AI scoring breakdown
            </div>
            <div className="flex items-center text-sm text-gray-300">
              <span className="text-green-400 mr-2">✓</span>
              Niche-specific performance model
            </div>
            <div className="flex items-center text-sm text-gray-300">
              <span className="text-green-400 mr-2">✓</span>
              Priority processing
            </div>
          </div>

          {/* Buttons */}
          <div className="flex flex-col gap-3">
            <button
              onClick={handleJoinWaitlist}
              className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg font-semibold transition-all duration-200"
            >
              Join Creator Waitlist
            </button>
            
            <p className="text-xs text-gray-400 text-center">
              Launching in 2-3 weeks • $19/month • 25 analyses
            </p>
            
            <Link
              href="/pricing"
              className="text-sm text-gray-400 hover:text-gray-300 transition text-center"
            >
              View all plans →
            </Link>
          </div>

        </div>
      </div>
      
      {/* Waitlist Modal */}
      <WaitlistModal
        isOpen={showWaitlist}
        onClose={() => setShowWaitlist(false)}
        plan="creator"
      />
    </div>
  )
}
