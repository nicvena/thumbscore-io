'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { WaitlistModal } from '@/app/components/WaitlistModal'

export default function PricingPage() {
  const [showWaitlist, setShowWaitlist] = useState(false)
  const [selectedPlan, setSelectedPlan] = useState<'creator' | 'pro'>('creator')
  const [waitlistCount, setWaitlistCount] = useState(0)
  
  useEffect(() => {
    // Fetch waitlist count
    fetch('/api/waitlist')
      .then(r => r.json())
      .then(data => setWaitlistCount(data.count))
      .catch(() => setWaitlistCount(47)) // Fallback count for demo
  }, [])
  
  function handleJoinWaitlist(plan: 'creator' | 'pro') {
    setSelectedPlan(plan)
    setShowWaitlist(true)
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 py-20">
      <div className="container mx-auto px-4">
        {/* Back to Home Button */}
        <div className="mb-8">
          <Link 
            href="/" 
            className="inline-flex items-center text-blue-400 hover:text-blue-300 text-sm transition-colors duration-200"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Home
          </Link>
        </div>

        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-white mb-6">Choose Your Plan</h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-8">
            Unlock AI-powered thumbnail optimization with our globally-ready pricing tiers.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <Plan 
            title="Free" 
            price="$0" 
            period="/month"
            features={[
              "1 thumbnail analysis per month",
              "Smart AI score + winner explanation",
              "No signup required"
            ]} 
            button="Try It Free" 
            popular={false}
            disabled={true}
          />
          
          <Plan 
            title="Creator" 
            price="$19" 
            period="/month"
            features={[
              "25 analyses per month",
              "Full AI scoring breakdown (clarity, emotion, composition)",
              "Niche-specific performance model",
              "Priority processing"
            ]} 
            button="Join Waitlist" 
            popular={true}
            badge="Best for YouTubers"
            waitlistCount={waitlistCount}
            onJoinWaitlist={() => handleJoinWaitlist('creator')}
          />
          
          <Plan 
            title="Pro" 
            price="$49" 
            period="/month"
            features={[
              "100 analyses per month",
              "All Creator features",
              "Compare historical results (A/B mode)",
              "Early access to new AI models"
            ]} 
            button="Join Waitlist" 
            popular={false}
            onJoinWaitlist={() => handleJoinWaitlist('pro')}
          />
        </div>

        <div className="text-center mt-16">
          <p className="text-gray-400 text-sm">
            Launching soon • All plans include our proprietary AI analysis engine
          </p>
        </div>
        
        {/* Waitlist Modal */}
        <WaitlistModal
          isOpen={showWaitlist}
          onClose={() => setShowWaitlist(false)}
          plan={selectedPlan}
        />
      </div>
    </div>
  )
}

function Plan({ title, price, period, features, button, stripePriceId, popular, disabled, badge, waitlistCount, onJoinWaitlist }: {
  title: string
  price: string
  period: string
  features: string[]
  button: string
  stripePriceId?: string
  popular?: boolean
  disabled?: boolean
  badge?: string
  waitlistCount?: number
  onJoinWaitlist?: () => void
}) {
  const [isLoading, setIsLoading] = useState(false)

  function handleClick() {
    if (disabled) return
    
    if (onJoinWaitlist) {
      onJoinWaitlist()
    } else if (stripePriceId) {
      handleCheckout()
    }
  }

  async function handleCheckout() {
    if (!stripePriceId || disabled) return
    
    setIsLoading(true)
    try {
      const res = await fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priceId: stripePriceId })
      })
      const data = await res.json()
      
      if (data.url) {
        window.location.href = data.url
      } else {
        console.error('No checkout URL received')
      }
    } catch (error) {
      console.error('Error creating checkout session:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={`relative p-8 rounded-2xl bg-gradient-to-b from-gray-800 to-gray-900 border-2 transition-all duration-300 hover:scale-105 hover:shadow-lg ${
      popular 
        ? 'border-purple-500 shadow-2xl shadow-purple-500/20' 
        : disabled
        ? 'border-gray-600 opacity-75'
        : 'border-gray-700 hover:border-gray-600'
    }`}>
      {popular && (
        <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
          <span className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-4 py-1 rounded-full text-sm font-semibold shadow-lg">
            {badge || 'Most Popular'}
          </span>
        </div>
      )}
      
      <div className="text-center">
        <h3 className="text-3xl font-bold text-white mb-2">{title}</h3>
        <div className="mb-6">
          <div className="flex items-baseline justify-center">
            <span className="text-5xl font-bold text-purple-400">{price}</span>
            <span className="text-lg text-gray-400 ml-1">{period}</span>
          </div>
        </div>
        
        <ul className="text-gray-300 text-left mb-8 space-y-3">
          {features.map((feature, index) => (
            <li key={index} className="flex items-start">
              <span className="text-green-400 mr-3 mt-0.5 flex-shrink-0">✓</span>
              <span className="text-sm leading-relaxed">{feature}</span>
            </li>
          ))}
        </ul>
        
        {/* Waitlist Badge */}
        {onJoinWaitlist && waitlistCount && (
          <div className="text-center mb-4">
            <div className="inline-flex items-center px-3 py-1 bg-purple-600/20 border border-purple-500/30 rounded-full text-sm text-purple-300">
              🎯 {waitlistCount}+ creators waiting
            </div>
          </div>
        )}
        
        {/* Launch Status */}
        {onJoinWaitlist && (
          <div className="text-center mb-4">
            <span className="text-xs text-gray-400">Launching in 2-3 weeks</span>
          </div>
        )}
        
        <button
          onClick={handleClick}
          disabled={disabled || isLoading}
          className={`w-full py-4 rounded-lg font-semibold text-lg transition-all duration-200 ${
            disabled
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : popular
              ? 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-lg hover:shadow-xl'
              : 'bg-gray-700 hover:bg-gray-600 text-white hover:shadow-lg'
          } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isLoading ? 'Processing...' : button}
        </button>
      </div>
    </div>
  )
}