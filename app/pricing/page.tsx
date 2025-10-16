'use client';

import Link from 'next/link';
import { useState } from 'react';
import { useAnalytics } from '@/lib/hooks/useAnalytics';

interface PricingTier {
  name: string;
  monthlyPrice: number;
  annualPrice: number;
  description: string;
  features: string[];
  limitations?: string[];
  buttonText: string;
  buttonAction: string;
  popular?: boolean;
  priceId?: {
    monthly: string;
    annual: string;
  };
}

const pricingTiers: PricingTier[] = [
  {
    name: 'Free',
    monthlyPrice: 0,
    annualPrice: 0,
    description: 'Perfect for testing the waters',
    features: [
      '5 thumbnail analyses per month',
      'Basic AI scoring (visual, text, contrast, emotion)',
      'Community support',
      'No credit card required'
    ],
    buttonText: 'Get Started Free',
    buttonAction: '/upload'
  },
  {
    name: 'Creator',
    monthlyPrice: 19,
    annualPrice: 15,
    description: 'Best for growing creators',
    features: [
      '100 thumbnail analyses per month',
      'Advanced AI scoring with full breakdown',
      'A/B testing history (30 days)',
      'Trend analysis dashboard',
      'Email support',
      'Export reports (PDF)',
      '14-day free trial'
    ],
    buttonText: 'Start Free Trial',
    buttonAction: 'subscribe-creator',
    popular: true,
    priceId: {
      monthly: 'price_creator_monthly',
      annual: 'price_creator_annual'
    }
  },
  {
    name: 'Pro',
    monthlyPrice: 49,
    annualPrice: 39,
    description: 'For agencies and power users',
    features: [
      'Unlimited thumbnail analyses',
      'Premium AI scoring engine',
      'Competitor benchmarking',
      'Unlimited A/B testing history',
      'Custom niche training',
      'Priority 24/7 support',
      'White-label reports',
      'Full API access',
      '14-day free trial'
    ],
    buttonText: 'Start Free Trial',
    buttonAction: 'subscribe-pro',
    priceId: {
      monthly: 'price_pro_monthly',
      annual: 'price_pro_annual'
    }
  }
];

export default function PricingPage() {
  const [isAnnual, setIsAnnual] = useState(false);
  const analytics = useAnalytics();

  const handlePricingToggle = (annual: boolean) => {
    setIsAnnual(annual);
    analytics.trackEvent('pricing_toggle', {
      event_category: 'engagement',
      event_label: annual ? 'annual' : 'monthly',
    });
  };

  const handleCtaClick = (tier: string, action: string) => {
    analytics.trackEvent('pricing_cta_click', {
      event_category: 'conversion',
      event_label: tier,
      custom_parameter_1: action,
    });

    if (action.startsWith('subscribe-')) {
      // TODO: Integrate with Stripe
      console.log(`Subscribing to ${tier} plan`);
      alert(`Stripe integration coming soon! ${tier} plan selected.`);
    }
  };

  const getPrice = (tier: PricingTier) => {
    return isAnnual ? tier.annualPrice : tier.monthlyPrice;
  };

  const getSavings = (tier: PricingTier) => {
    if (tier.monthlyPrice === 0) return 0;
    const annualTotal = tier.annualPrice * 12;
    const monthlyTotal = tier.monthlyPrice * 12;
    return monthlyTotal - annualTotal;
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] text-white">
      <div className="max-w-7xl mx-auto px-6 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm mb-4 inline-block">
            ← Back to Home
          </Link>
          <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-[#6a5af9] via-[#1de9b6] to-[#6a5af9] bg-clip-text text-transparent">
            Simple Pricing. Start Free.
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            No credit card required. Upgrade when you're ready.
          </p>

          {/* Billing Toggle */}
          <div className="flex items-center justify-center gap-4 mb-8">
            <span className={`text-lg ${!isAnnual ? 'text-white font-semibold' : 'text-gray-400'}`}>
              Monthly
            </span>
            <button
              onClick={() => handlePricingToggle(!isAnnual)}
              className={`relative w-16 h-8 rounded-full transition-colors ${
                isAnnual ? 'bg-gradient-to-r from-[#6a5af9] to-[#1de9b6]' : 'bg-gray-600'
              }`}
            >
              <div className={`absolute top-1 left-1 w-6 h-6 bg-white rounded-full transition-transform ${
                isAnnual ? 'translate-x-8' : 'translate-x-0'
              }`} />
            </button>
            <span className={`text-lg ${isAnnual ? 'text-white font-semibold' : 'text-gray-400'}`}>
              Annual
            </span>
            {isAnnual && (
              <span className="bg-green-600 text-white text-sm px-3 py-1 rounded-full font-semibold">
                Save 20%
              </span>
            )}
          </div>
        </div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {pricingTiers.map((tier, index) => (
            <div
              key={tier.name}
              className={`relative bg-white/5 backdrop-blur-sm rounded-xl p-8 border transition-all hover:scale-105 ${
                tier.popular
                  ? 'border-[#6a5af9] shadow-lg shadow-[#6a5af9]/20'
                  : 'border-white/10 hover:border-cyan-500/50'
              }`}
            >
              {tier.popular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <span className="bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white px-4 py-2 rounded-full text-sm font-semibold">
                    Most Popular
                  </span>
                </div>
              )}

              {/* Plan Name */}
              <div className="text-center mb-6">
                <h3 className="text-2xl font-bold mb-2">{tier.name}</h3>
                <p className="text-gray-400">{tier.description}</p>
              </div>

              {/* Price */}
              <div className="text-center mb-8">
                <div className="flex items-baseline justify-center gap-1">
                  <span className="text-4xl font-bold">${getPrice(tier)}</span>
                  {tier.monthlyPrice > 0 && (
                    <span className="text-gray-400">/{isAnnual ? 'month' : 'month'}</span>
                  )}
                </div>
                {isAnnual && tier.monthlyPrice > 0 && (
                  <div className="mt-2">
                    <span className="text-sm text-gray-400 line-through">
                      ${tier.monthlyPrice}/month
                    </span>
                    <span className="text-sm text-green-400 ml-2">
                      Save ${getSavings(tier)}/year
                    </span>
                  </div>
                )}
                {isAnnual && tier.monthlyPrice > 0 && (
                  <p className="text-xs text-gray-500 mt-1">
                    Billed annually (${getPrice(tier) * 12}/year)
                  </p>
                )}
              </div>

              {/* Features */}
              <div className="mb-8">
                <ul className="space-y-3">
                  {tier.features.map((feature, featureIndex) => (
                    <li key={featureIndex} className="flex items-start gap-3">
                      <span className="text-green-400 mt-0.5">✓</span>
                      <span className="text-gray-300">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* CTA Button */}
              <div className="mt-auto">
                {tier.buttonAction.startsWith('/') ? (
                  <Link
                    href={tier.buttonAction}
                    onClick={() => handleCtaClick(tier.name.toLowerCase(), tier.buttonAction)}
                    className={`block w-full py-4 px-6 rounded-lg font-semibold text-center transition-all ${
                      tier.popular
                        ? 'bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white hover:shadow-lg hover:shadow-cyan-500/50'
                        : tier.name === 'Free'
                        ? 'bg-white/10 text-white hover:bg-white/20'
                        : 'bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white hover:shadow-lg hover:shadow-cyan-500/50'
                    }`}
                  >
                    {tier.buttonText}
                  </Link>
                ) : (
                  <button
                    onClick={() => handleCtaClick(tier.name.toLowerCase(), tier.buttonAction)}
                    className={`block w-full py-4 px-6 rounded-lg font-semibold text-center transition-all ${
                      tier.popular
                        ? 'bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white hover:shadow-lg hover:shadow-cyan-500/50'
                        : 'bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white hover:shadow-lg hover:shadow-cyan-500/50'
                    }`}
                  >
                    {tier.buttonText}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* FAQ Section */}
        <div className="mt-20 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Frequently Asked Questions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2 text-cyan-300">Can I cancel anytime?</h3>
                <p className="text-gray-400">Yes, you can cancel your subscription at any time. No long-term contracts or cancellation fees.</p>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2 text-cyan-300">What's included in the free trial?</h3>
                <p className="text-gray-400">Full access to all features of your chosen plan for 14 days. No credit card required to start.</p>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2 text-cyan-300">Do you offer refunds?</h3>
                <p className="text-gray-400">Yes, we offer a 30-day money-back guarantee if you're not satisfied with your subscription.</p>
              </div>
            </div>
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-2 text-cyan-300">How reliable are the results?</h3>
                <p className="text-gray-400">Very reliable. ThumbScore provides consistent results - the same thumbnail will score within ±2 points every time, giving you confidence in your decisions.</p>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2 text-cyan-300">Can I upgrade or downgrade?</h3>
                <p className="text-gray-400">Yes, you can change your plan at any time. Changes take effect immediately with prorated billing.</p>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2 text-cyan-300">Is there an API available?</h3>
                <p className="text-gray-400">Yes, Pro plan includes full API access for integrating thumbnail scoring into your own tools and workflows.</p>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        <div className="mt-16 text-center bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-lg p-8 border border-white/10">
          <h2 className="text-3xl font-bold mb-4">Ready to Stop Guessing?</h2>
          <p className="text-xl text-gray-300 mb-8">
            Join thousands of creators who've improved their thumbnail performance with AI-powered insights.
          </p>
          <div className="flex gap-4 justify-center">
            <Link
              href="/upload"
              className="px-8 py-4 bg-gradient-to-r from-[#6a5af9] to-[#1de9b6] text-white rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all text-lg font-semibold"
            >
              Try Free Now
            </Link>
            <Link
              href="/faq"
              className="px-8 py-4 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors text-lg font-semibold"
            >
              Learn More
            </Link>
          </div>
        </div>

        {/* Footer Links */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <Link href="/" className="hover:text-gray-300 mx-2">Home</Link>
          <span>•</span>
          <Link href="/upload" className="hover:text-gray-300 mx-2">Upload</Link>
          <span>•</span>
          <Link href="/faq" className="hover:text-gray-300 mx-2">FAQ</Link>
          <span>•</span>
          <a href="mailto:support@thumbscore.io" className="hover:text-gray-300 mx-2">Contact</a>
        </div>
      </div>
    </main>
  );
}