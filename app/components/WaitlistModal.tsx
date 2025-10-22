'use client'

import { useState } from 'react'

interface WaitlistModalProps {
  isOpen: boolean
  onClose: () => void
  plan: 'creator' | 'pro'
}

export function WaitlistModal({ isOpen, onClose, plan }: WaitlistModalProps) {
  const [email, setEmail] = useState('')
  const [maxPrice, setMaxPrice] = useState('$19')
  const [interests, setInterests] = useState<string[]>([])
  const [status, setStatus] = useState<'idle' | 'loading' | 'success'>('idle')
  
  const planDetails = {
    creator: {
      name: 'Creator Plan',
      price: '$19/month',
      analyses: '25 analyses/month'
    },
    pro: {
      name: 'Pro Plan',
      price: '$49/month',
      analyses: '100 analyses/month'
    }
  }
  
  const currentPlan = planDetails[plan]
  
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setStatus('loading')
    
    try {
      const res = await fetch('/api/waitlist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          plan,
          maxPrice,
          interests
        })
      })
      
      if (res.ok) {
        setStatus('success')
        // Close modal after 3 seconds
        setTimeout(() => {
          onClose()
          setStatus('idle')
          setEmail('')
          setMaxPrice('$19')
          setInterests([])
        }, 3000)
      } else {
        setStatus('idle')
        alert('Failed to join waitlist. Please try again.')
      }
    } catch (error) {
      console.error('Waitlist signup failed:', error)
      setStatus('idle')
      alert('Failed to join waitlist. Please try again.')
    }
  }
  
  if (!isOpen) return null
  
  if (status === 'success') {
    return (
      <div className='fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4'>
        <div className='bg-gray-800 rounded-2xl p-8 max-w-md mx-4 text-center'>
          <div className='text-6xl mb-4'>🎉</div>
          <h2 className='text-2xl font-bold text-white mb-2'>
            You're on the Waitlist!
          </h2>
          <p className='text-gray-300 mb-4'>
            We'll email you when {currentPlan.name} launches.
          </p>
          <p className='text-sm text-gray-400'>
            Expected launch: 2-3 weeks
          </p>
        </div>
      </div>
    )
  }
  
  return (
    <div className='fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4'>
      <div className='bg-gray-800 rounded-2xl p-8 max-w-md w-full relative max-h-[90vh] overflow-y-auto'>
        <button
          onClick={onClose}
          className='absolute top-4 right-4 text-gray-400 hover:text-white text-xl'
        >
          ✕
        </button>
        
        <h2 className='text-2xl font-bold text-white mb-2'>
          Join {currentPlan.name} Waitlist
        </h2>
        <p className='text-gray-400 mb-6'>
          Be first to know when we launch. Help us prioritize features.
        </p>
        
        <form onSubmit={handleSubmit} className='space-y-4'>
          {/* Email */}
          <div>
            <label className='block text-sm font-medium text-gray-300 mb-2'>
              Email
            </label>
            <input
              type='email'
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className='w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500'
              placeholder='your@email.com'
            />
          </div>
          
          {/* Max Price Willing to Pay */}
          <div>
            <label className='block text-sm font-medium text-gray-300 mb-2'>
              What would you pay monthly?
            </label>
            <select
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
              className='w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500'
            >
              <option value='$9'>$9/month</option>
              <option value='$19'>$19/month</option>
              <option value='$29'>$29/month</option>
              <option value='$49'>$49/month</option>
            </select>
          </div>
          
          {/* Feature Interests */}
          <div>
            <label className='block text-sm font-medium text-gray-300 mb-2'>
              What features interest you? (optional)
            </label>
            <div className='space-y-2'>
              {[
                'Competitor comparison',
                'Historical tracking',
                'Team accounts',
                'API access',
                'More analyses per month'
              ].map((feature) => (
                <label key={feature} className='flex items-center'>
                  <input
                    type='checkbox'
                    checked={interests.includes(feature)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setInterests([...interests, feature])
                      } else {
                        setInterests(interests.filter(i => i !== feature))
                      }
                    }}
                    className='mr-2'
                  />
                  <span className='text-gray-300 text-sm'>{feature}</span>
                </label>
              ))}
            </div>
          </div>
          
          {/* Plan Summary */}
          <div className='bg-gray-700/50 rounded-lg p-4 border border-gray-600'>
            <div className='text-sm text-gray-400 mb-1'>{currentPlan.name}</div>
            <div className='text-lg font-bold text-white'>{currentPlan.price}</div>
            <div className='text-sm text-gray-400'>{currentPlan.analyses}</div>
          </div>
          
          {/* Submit Button */}
          <button
            type='submit'
            disabled={status === 'loading'}
            className='w-full py-3 bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-lg transition disabled:opacity-50'
          >
            {status === 'loading' ? 'Joining...' : 'Join Waitlist'}
          </button>
          
          <p className='text-xs text-gray-400 text-center'>
            We'll email you when spots open (targeting 2-3 weeks)
          </p>
        </form>
      </div>
    </div>
  )
}