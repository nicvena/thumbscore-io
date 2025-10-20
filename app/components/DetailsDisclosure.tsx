import React, { useState } from 'react';

interface TechnicalDetails {
  subjectCoveragePct?: number;
  textReadability?: number;
  contrast?: number;
  saturation?: number;
  ocrSample?: string;
  niche?: string;
  confidence?: number;
}

interface DetailsDisclosureProps {
  technical?: TechnicalDetails;
}

export default function DetailsDisclosure({ technical }: DetailsDisclosureProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!technical) return null;

  const formatMetric = (value: number | undefined, suffix: string = '') => {
    if (value === undefined) return 'N/A';
    return `${Math.round(value)}${suffix}`;
  };

  const formatSaturation = (value: number | undefined) => {
    if (value === undefined) return 'N/A';
    // Normalize to 0-100 if it's between 0-1
    const normalized = value <= 1 ? value * 100 : value;
    return `${Math.round(normalized)}%`;
  };

  return (
    <details 
      className="bg-gray-800/20 rounded-lg border border-gray-700/30 overflow-hidden"
      open={isOpen}
      onToggle={(e) => setIsOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary className="px-4 py-3 cursor-pointer hover:bg-gray-700/20 transition-colors flex items-center gap-2 text-sm font-medium text-gray-400">
        <span>🔍</span>
        <span>View technical details</span>
        <span className="ml-auto text-xs text-gray-500">
          {isOpen ? '▼' : '▶'}
        </span>
      </summary>
      
      <div className="px-4 pb-4 border-t border-gray-700/30">
        <div className="grid grid-cols-2 gap-4 pt-4 text-sm">
          {technical.subjectCoveragePct !== undefined && (
            <div>
              <div className="text-gray-400">Subject Coverage</div>
              <div className="text-gray-200 font-medium">{formatMetric(technical.subjectCoveragePct, '%')}</div>
            </div>
          )}
          
{/* Text Readability metric removed to avoid exposing unreliable text metrics */}
          
          {technical.contrast !== undefined && (
            <div>
              <div className="text-gray-400">Contrast Score</div>
              <div className="text-gray-200 font-medium">{formatMetric(technical.contrast, '%')}</div>
            </div>
          )}
          
          {technical.saturation !== undefined && (
            <div>
              <div className="text-gray-400">Color Saturation</div>
              <div className="text-gray-200 font-medium">{formatSaturation(technical.saturation)}</div>
            </div>
          )}
          
          {technical.confidence !== undefined && (
            <div>
              <div className="text-gray-400">Confidence</div>
              <div className="text-gray-200 font-medium">{formatMetric(technical.confidence, '%')}</div>
            </div>
          )}
          
          {technical.niche && (
            <div>
              <div className="text-gray-400">Content Category</div>
              <div className="text-gray-200 font-medium capitalize">{technical.niche}</div>
            </div>
          )}
        </div>

{/* Detected Text section removed to avoid showing unreliable OCR results */}

        <div className="mt-4 pt-3 border-t border-gray-700/30">
          <p className="text-xs text-gray-500 text-center">
            Metrics are estimates derived from our vision pipeline.
          </p>
        </div>
      </div>
    </details>
  );
}
