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
    question: "How does Thumbnail Lab work?",
    answer: "Upload up to 3 thumbnail options, and our analysis engine scores them based on proven thumbnail principles: text clarity, subject prominence, color contrast, emotional appeal, composition, and title alignment. We analyze patterns from 120,000+ YouTube thumbnails to predict which will get more clicks."
  },
  {
    question: "Is it really free?",
    answer: "Yes! Thumbnail Lab is completely free to use. No signup required, no credit card needed. Just upload your thumbnails and get instant analysis."
  },
  {
    question: "How accurate are the predictions?",
    answer: "Our scoring is based on data-backed patterns from 120,000+ analyzed YouTube thumbnails across 6 niches. The Pattern Coach shows specific techniques that have delivered 20-50% CTR improvements in real channels. While we can't guarantee results (YouTube success depends on many factors), our recommendations are grounded in proven thumbnail principles."
  },
  {
    question: "What makes a thumbnail score high?",
    answer: "High-scoring thumbnails typically have: 1-3 bold words with high contrast, large subject/face (30-40% of frame), vibrant colors, clear emotion, simple composition, and strong alignment with the video title. Our Pattern Coach shows niche-specific patterns that work best."
  },
  {
    question: "Can I save my analysis results?",
    answer: "Results are available via a unique link for 7 days. Bookmark the results page or copy the link to access later. We're considering adding user accounts based on feedback - let us know if you'd like this feature!"
  },
  {
    question: "What image formats are supported?",
    answer: "We support JPG, PNG, WebP, and most common image formats. For best results, use YouTube's recommended thumbnail size: 1280×720 pixels (16:9 aspect ratio)."
  },
  {
    question: "Why do I need to upload 3 thumbnails?",
    answer: "Comparing multiple options helps you understand what makes a thumbnail effective. You can see the difference in scores and get specific recommendations for each variation. If you only have one, upload it 3 times to get detailed insights!"
  },
  {
    question: "What are the 'sub-scores'?",
    answer: "We break down the overall CTR score into 6 components: Clarity (text readability), Subject Prominence (face/object size), Contrast Pop (color vibrancy), Emotion (expression intensity), Visual Hierarchy (composition), and Title Match (alignment with video title). This helps you understand exactly what to improve."
  },
  {
    question: "What is the Pattern Coach?",
    answer: "Pattern Coach shows data-backed thumbnail patterns from 10,000+ videos in your niche (education, gaming, tech, etc.). Each pattern includes the CTR improvement percentage based on real data. For example, 'Fewer Words + Big Face' shows +42% CTR lift in education videos."
  },
  {
    question: "How do the visual overlays work?",
    answer: "Click the overlay buttons to see: Saliency Heatmap (where viewers look), OCR Boxes (text readability), Face Boxes (emotion detection), and Rule of Thirds Grid (composition). These help you understand WHY a thumbnail scores high or low."
  },
  {
    question: "Can I download the improved thumbnails?",
    answer: "Auto-fix functionality is coming soon! For now, use our specific recommendations to manually improve your thumbnails in your favorite image editor. We provide exact guidance like 'Increase subject size by 25%' or 'Reduce to 2-3 words'."
  },
  {
    question: "Does it work for all YouTube niches?",
    answer: "Yes! We have specific patterns for Tech, Gaming, Education, Entertainment, Beauty, and News. If your niche isn't listed, we use general patterns from 120,000+ thumbnails that work across all categories."
  },
  {
    question: "Is my data private?",
    answer: "We don't store your uploaded images permanently. Thumbnails are processed in memory and discarded after analysis. Session data expires after 7 days. We don't collect personal information or require accounts."
  },
  {
    question: "Can I use this for commercial purposes?",
    answer: "Absolutely! Whether you're a solo creator, agency, or business, feel free to use Thumbnail Lab to optimize your YouTube thumbnails. No attribution required."
  },
  {
    question: "How is this different from other thumbnail tools?",
    answer: "Most tools just give vague advice. Thumbnail Lab provides: specific CTR scores, data-backed recommendations (not opinions), visual overlays showing exactly what to fix, and niche-specific patterns with real CTR lift percentages. Plus, it's completely free!"
  },
  {
    question: "Will you add more features?",
    answer: "Yes! We're considering: user accounts, analysis history, A/B test tracking, auto-fix downloads, and real-time AI models. Vote for features you want by providing feedback after your analysis!"
  },
  {
    question: "Can I integrate this into my app?",
    answer: "Yes! We have a REST API at POST /api/v1/score. Send thumbnail URLs and get back scores, insights, and recommendations. Perfect for automation or building your own tools. Check our API documentation for details."
  },
  {
    question: "What if the analysis seems wrong?",
    answer: "Our scoring is based on general patterns that work across YouTube. Your specific audience might respond differently. Use our insights as a starting point, then A/B test on YouTube to see what actually works for your channel. We're always improving - send feedback to help us get better!"
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
            Everything you need to know about Thumbnail Lab
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
