/**
 * FAQ Page
 * Answers common questions about Thumbnail Lab
 */

'use client';

import Link from 'next/link';
import { useState } from 'react';

interface FAQItem {
  question: string;
  answer: string;
}

const faqs: FAQItem[] = [
  {
    question: "How does ThumbScore work?",
    answer: "ThumbScore uses advanced AI to analyze your thumbnails like a human expert would - looking at text clarity, visual contrast, emotional appeal, and niche-specific conventions. Results in 10 seconds."
  },
  {
    question: "Is it accurate?",
    answer: "We've tested ThumbScore extensively to ensure consistent, reliable results. The same thumbnail analyzed multiple times will score within 2 points - that's the consistency you need to make confident decisions."
  },
  {
    question: "Does it work for my niche?",
    answer: "Yes! ThumbScore has specialized intelligence for 10 content categories: Gaming, Business, Tech, Cooking, Fitness, Education, Travel, Music, Entertainment, and General. Each niche gets advice tailored to what actually works for that type of content."
  },
  {
    question: "Why is it so fast?",
    answer: "Unlike A/B testing (48 hours) or human feedback (hours/days), ThumbScore's AI analyzes thumbnails in seconds. No waiting, no delays - just instant answers."
  },
  {
    question: "Can I try it for free?",
    answer: "Yes! Every account gets 5 free analyses per month. No credit card required. Upgrade to Creator ($19/month) or Pro ($49/month) when you're ready for more."
  },
  {
    question: "What if I don't like the results?",
    answer: "All paid plans include a 14-day free trial. Cancel anytime, no questions asked."
  },
  {
    question: "What image formats are supported?",
    answer: "We support JPG, PNG, WebP, and most common image formats. For best results, use YouTube's recommended thumbnail size: 1280×720 pixels (16:9 aspect ratio)."
  },
  {
    question: "How reliable are the scores?",
    answer: "Very reliable. ThumbScore provides consistent results - the same thumbnail will score within ±2 points every time. This consistency lets you make confident decisions about which thumbnail to use."
  },
  {
    question: "What makes this different from other tools?",
    answer: "Most tools give generic advice. ThumbScore provides niche-specific analysis trained on 10 different content categories. Gaming thumbnails are analyzed differently than business thumbnails because they work differently."
  },
  {
    question: "Can I use this for commercial purposes?",
    answer: "Absolutely! Whether you're a solo creator, agency, or business, ThumbScore helps you choose winning thumbnails. Many creators and agencies use it daily."
  },
  {
    question: "How do I choose which thumbnail to use?",
    answer: "Simple - upload your 3 options, and ThumbScore tells you which one gets the highest score. Use that one. The AI handles all the complex analysis for you."
  },
  {
    question: "Is my data private?",
    answer: "Yes. We process your thumbnails for analysis but don't store them permanently. Your images are analyzed and then discarded. We don't collect personal information."
  },
  {
    question: "What's included in the analysis?",
    answer: "You get an overall score (30-95 range), breakdown of what's working, specific suggestions for improvement, and clear guidance on which thumbnail performs best."
  },
  {
    question: "Can I get API access?",
    answer: "Yes! We have a REST API for developers. Perfect for automation or building ThumbScore into your own tools. Contact us for API documentation."
  },
  {
    question: "Do you have bulk analysis?",
    answer: "Pro plans include bulk analysis features. Perfect for agencies or creators who need to analyze many thumbnails quickly."
  },
  {
    question: "How often should I test thumbnails?",
    answer: "Test every video! The 10-second analysis time makes it easy to test 3 options for every upload. Many creators see this as essential as spell-checking their titles."
  }
];

export default function FAQPage() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggleQuestion = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <main className="min-h-screen bg-black text-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        {/* Header */}
        <div className="text-center mb-12">
          <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm mb-4 inline-block">
            ← Back to Home
          </Link>
          <h1 className="text-4xl font-bold mb-4">Frequently Asked Questions</h1>
          <p className="text-gray-400">
            Everything you need to know about ThumbScore
          </p>
        </div>

        {/* FAQ Accordion */}
        <div className="space-y-3">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden transition-all"
            >
              <button
                onClick={() => toggleQuestion(index)}
                className="w-full flex items-center justify-between p-5 text-left hover:bg-gray-750 transition-colors"
              >
                <h3 className="text-lg font-semibold text-white pr-4">
                  {faq.question}
                </h3>
                <span className="text-2xl text-gray-400 flex-shrink-0 transform transition-transform">
                  {openIndex === index ? '−' : '+'}
                </span>
              </button>
              
              {openIndex === index && (
                <div className="px-5 pb-5 text-gray-300 leading-relaxed animate-fade-in">
                  {faq.answer}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Still have questions? */}
        <div className="mt-12 text-center bg-gradient-to-r from-blue-900 to-purple-900 rounded-lg p-8">
          <h2 className="text-2xl font-bold mb-3">Still have questions?</h2>
          <p className="text-gray-300 mb-6">
            We&apos;re here to help! Try the app and send us feedback.
          </p>
          <div className="flex gap-4 justify-center">
            <Link
              href="/upload"
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
            >
              Try It Free
            </Link>
            <a
              href="mailto:support@thumbnaillab.com"
              className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold transition-colors"
            >
              Contact Us
            </a>
          </div>
        </div>

        {/* Quick Links */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <Link href="/" className="hover:text-gray-300 mx-2">Home</Link>
          <span>•</span>
          <Link href="/upload" className="hover:text-gray-300 mx-2">Upload</Link>
          <span>•</span>
          <Link href="/pricing" className="hover:text-gray-300 mx-2">Pricing</Link>
          <span>•</span>
          <a href="/api/v1/score" className="hover:text-gray-300 mx-2">API Docs</a>
        </div>
      </div>

      <style jsx global>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fade-in {
          animation: fade-in 0.3s ease-out;
        }
      `}</style>
    </main>
  );
}
