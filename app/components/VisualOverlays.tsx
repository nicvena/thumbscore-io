/**
 * Visual Overlays Component
 * 
 * Renders interactive visualization overlays:
 * - Saliency heatmap (attention prediction)
 * - OCR boxes (text detection)
 * - Face boxes (emotion analysis)
 * - Rule of thirds grid
 */

'use client';

import { useState } from 'react';

interface OverlayProps {
  thumbnailId: number;
  fileName: string;
  imageUrl?: string;
  width?: number;
  height?: number;
  heatmapData?: Array<{ x: number; y: number; intensity: number; label: string }>;
  ocrBoxes?: Array<{ text: string; bbox: number[]; confidence: number }>;
  faceBoxes?: Array<{ bbox: number[]; emotion: string; confidence: number }>;
}

export default function VisualOverlays({
  thumbnailId,
  fileName,
  imageUrl,
  width = 1280,
  height = 720,
  heatmapData = [],
  ocrBoxes = [],
  faceBoxes = []
}: OverlayProps) {
  const [activeOverlay, setActiveOverlay] = useState<'none' | 'saliency' | 'ocr' | 'faces' | 'thirds'>('none');

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      {/* Overlay Toggle Buttons */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <button
          onClick={() => setActiveOverlay(activeOverlay === 'saliency' ? 'none' : 'saliency')}
          className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
            activeOverlay === 'saliency'
              ? 'bg-purple-600 text-white ring-2 ring-purple-400'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          {activeOverlay === 'saliency' && '✓ '}🔥 Heatmap
        </button>
        <button
          onClick={() => setActiveOverlay(activeOverlay === 'ocr' ? 'none' : 'ocr')}
          className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
            activeOverlay === 'ocr'
              ? 'bg-blue-600 text-white ring-2 ring-blue-400'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          {activeOverlay === 'ocr' && '✓ '}📝 OCR
        </button>
        <button
          onClick={() => setActiveOverlay(activeOverlay === 'faces' ? 'none' : 'faces')}
          className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
            activeOverlay === 'faces'
              ? 'bg-green-600 text-white ring-2 ring-green-400'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          {activeOverlay === 'faces' && '✓ '}😊 Faces
        </button>
        <button
          onClick={() => setActiveOverlay(activeOverlay === 'thirds' ? 'none' : 'thirds')}
          className={`px-3 py-2 rounded text-xs font-semibold transition-all ${
            activeOverlay === 'thirds'
              ? 'bg-yellow-600 text-white ring-2 ring-yellow-400'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          {activeOverlay === 'thirds' && '✓ '}📐 Grid
        </button>
      </div>

      {/* Visualization Canvas */}
      <div className="relative aspect-video bg-gray-800 rounded-lg overflow-hidden">
        {/* Base thumbnail image or placeholder */}
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={`Thumbnail ${thumbnailId}`}
            className="absolute inset-0 w-full h-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-gray-600">
            <div className="text-center">
              <div className="text-6xl mb-2">🖼️</div>
              <p className="text-sm">{fileName}</p>
            </div>
          </div>
        )}

        {/* Saliency Heatmap Overlay */}
        {activeOverlay === 'saliency' && (
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
            <defs>
              <radialGradient id={`heatmap-${thumbnailId}`}>
                <stop offset="0%" stopColor="rgba(255,0,0,0.6)" />
                <stop offset="50%" stopColor="rgba(255,255,0,0.3)" />
                <stop offset="100%" stopColor="rgba(0,0,255,0)" />
              </radialGradient>
            </defs>
            {heatmapData.map((point, i) => (
              <g key={i}>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={point.intensity * 20}
                  fill={`url(#heatmap-${thumbnailId})`}
                  opacity={point.intensity}
                />
                <text
                  x={point.x}
                  y={point.y + 5}
                  fill="white"
                  fontSize="3"
                  textAnchor="middle"
                  className="font-bold"
                  style={{ textShadow: '0 0 2px black' }}
                >
                  {Math.round(point.intensity * 100)}%
                </text>
              </g>
            ))}
          </svg>
        )}

        {/* OCR Boxes Overlay */}
        {activeOverlay === 'ocr' && (
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
            {ocrBoxes.map((box, i) => (
              <g key={i}>
                <rect
                  x={box.bbox[0]}
                  y={box.bbox[1]}
                  width={box.bbox[2] - box.bbox[0]}
                  height={box.bbox[3] - box.bbox[1]}
                  fill="none"
                  stroke={box.confidence > 0.8 ? '#22c55e' : '#ef4444'}
                  strokeWidth="0.5"
                  strokeDasharray="2,1"
                />
                <text
                  x={box.bbox[0] + 1}
                  y={box.bbox[1] - 1}
                  fill="white"
                  fontSize="3"
                  className="font-bold"
                  style={{ textShadow: '0 0 2px black' }}
                >
                  {box.text} ({Math.round(box.confidence * 100)}%)
                </text>
              </g>
            ))}
          </svg>
        )}

        {/* Face Boxes Overlay */}
        {activeOverlay === 'faces' && (
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
            {faceBoxes.map((face, i) => (
              <g key={i}>
                <rect
                  x={face.bbox[0]}
                  y={face.bbox[1]}
                  width={face.bbox[2] - face.bbox[0]}
                  height={face.bbox[3] - face.bbox[1]}
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth="0.7"
                />
                <rect
                  x={face.bbox[0]}
                  y={face.bbox[1] - 5}
                  width={15}
                  height={5}
                  fill="#22c55e"
                  rx="1"
                />
                <text
                  x={face.bbox[0] + 1}
                  y={face.bbox[1] - 1.5}
                  fill="black"
                  fontSize="2.5"
                  className="font-bold"
                >
                  {face.emotion} {Math.round(face.confidence * 100)}%
                </text>
              </g>
            ))}
          </svg>
        )}

        {/* Rule of Thirds Grid */}
        {activeOverlay === 'thirds' && (
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
            {/* Vertical lines */}
            <line x1="33.33" y1="0" x2="33.33" y2="100" stroke="#fbbf24" strokeWidth="0.3" strokeDasharray="2,2" />
            <line x1="66.67" y1="0" x2="66.67" y2="100" stroke="#fbbf24" strokeWidth="0.3" strokeDasharray="2,2" />
            {/* Horizontal lines */}
            <line x1="0" y1="33.33" x2="100" y2="33.33" stroke="#fbbf24" strokeWidth="0.3" strokeDasharray="2,2" />
            <line x1="0" y1="66.67" x2="100" y2="66.67" stroke="#fbbf24" strokeWidth="0.3" strokeDasharray="2,2" />
            {/* Intersection points */}
            {[
              { x: 33.33, y: 33.33 },
              { x: 66.67, y: 33.33 },
              { x: 33.33, y: 66.67 },
              { x: 66.67, y: 66.67 }
            ].map((point, i) => (
              <circle
                key={i}
                cx={point.x}
                cy={point.y}
                r="1.5"
                fill="#fbbf24"
                opacity="0.8"
              />
            ))}
            {/* Labels */}
            <text x="50" y="8" fill="#fbbf24" fontSize="3" textAnchor="middle" className="font-bold">
              Rule of Thirds Grid
            </text>
            <text x="50" y="96" fill="#fbbf24" fontSize="2.5" textAnchor="middle" opacity="0.7">
              Place key elements at intersection points
            </text>
          </svg>
        )}
      </div>

      {/* Overlay Legend */}
      {activeOverlay !== 'none' && (
        <div className="mt-3 text-xs text-gray-400 bg-gray-800 rounded p-3">
          {activeOverlay === 'saliency' && (
            <div>
              <p className="font-semibold text-white mb-1">🔥 Saliency Heatmap</p>
              <p>Shows predicted viewer attention. Red = high attention, Blue = low attention. Align key elements with hotspots.</p>
            </div>
          )}
          {activeOverlay === 'ocr' && (
            <div>
              <p className="font-semibold text-white mb-1">📝 OCR Contrast Analysis</p>
              <p>Green boxes = high readability ({'>'}80% contrast). Red boxes = poor readability ({'<'}80% contrast). Aim for 95%+ on all text.</p>
            </div>
          )}
          {activeOverlay === 'faces' && (
            <div>
              <p className="font-semibold text-white mb-1">😊 Face & Emotion Detection</p>
              <p>Green boxes show detected faces with emotion labels. Target: 25-40% of frame for optimal engagement.</p>
            </div>
          )}
          {activeOverlay === 'thirds' && (
            <div>
              <p className="font-semibold text-white mb-1">📐 Rule of Thirds Grid</p>
              <p>Position key elements (faces, text, objects) at intersection points (marked with circles) for better composition.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
