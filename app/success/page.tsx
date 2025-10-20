'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'

export default function SuccessPage() {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate processing time
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 2000)

    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <main className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex flex-col items-center justify-center p-6">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-purple-500 mx-auto mb-6"></div>
          <h1 className="text-3xl font-bold text-white mb-4">Setting up your account...</h1>
          <p className="text-gray-300">Please wait while we activate your subscription.</p>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex flex-col items-center justify-center p-6">
      <div className="text-center max-w-2xl mx-auto">
        <div className="mb-8">
          <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h1 className="text-4xl font-bold text-white mb-4">Welcome to Thumbscore! 🎉</h1>
          <p className="text-xl text-gray-300 mb-8">
            Your subscription is now active. You can now run unlimited AI thumbnail analyses.
          </p>
        </div>

        <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-2xl p-8 border border-purple-600/20 mb-8">
          <h2 className="text-2xl font-semibold text-white mb-4">What's Next?</h2>
          <div className="space-y-4 text-left">
            <div className="flex items-center">
              <span className="text-green-400 mr-3">✅</span>
              <span className="text-gray-300">Upload unlimited thumbnails for analysis</span>
            </div>
            <div className="flex items-center">
              <span className="text-green-400 mr-3">✅</span>
              <span className="text-gray-300">Get advanced AI insights and recommendations</span>
            </div>
            <div className="flex items-center">
              <span className="text-green-400 mr-3">✅</span>
              <span className="text-gray-300">Access priority support when needed</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            href="/upload"
            className="px-8 py-4 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-semibold transition-colors"
          >
            Start Analyzing →
          </Link>
          <Link
            href="/"
            className="px-8 py-4 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors"
          >
            Back to Home
          </Link>
        </div>

        <p className="text-sm text-gray-500 mt-8">
          Need help? Contact us at support@thumbscore.io
        </p>
      </div>
    </main>
  )
}
