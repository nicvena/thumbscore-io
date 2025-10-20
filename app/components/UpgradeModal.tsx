'use client'

import { useState } from 'react'
import Link from 'next/link'

interface UpgradeModalProps {
  isOpen: boolean
  onClose: () => void
}

export function UpgradeModal({ isOpen, onClose }: UpgradeModalProps) {
  const [isLoading, setIsLoading] = useState(false)

  if (!isOpen) return null

  const handleUpgrade = async () => {
    setIsLoading(true)
    // The upgrade will be handled by the pricing page
    window.location.href = '/pricing'
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
          
          <p className="text-gray-300 mb-6">
            You've used your free analysis. Upgrade to Creator for 25 analyses per month with advanced AI insights.
          </p>

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
          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleUpgrade}
              disabled={isLoading}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white rounded-lg font-semibold transition-all duration-200 disabled:opacity-50"
            >
              {isLoading ? 'Loading...' : 'Upgrade to Creator - $19/mo'}
            </button>
            
            <Link
              href="/signin"
              className="flex-1 px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors text-center"
            >
              Already Subscribed? Sign In
            </Link>
          </div>

        </div>
      </div>
    </div>
  )
}
