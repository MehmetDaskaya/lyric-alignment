'use client';

import { WordAlignment, AlignmentResult } from '@/lib/api';
import { cn } from '@/lib/utils';
import { X, Check } from 'lucide-react';

interface ComparisonViewProps {
  prediction: AlignmentResult;
  groundTruth?: AlignmentResult; // Optional standard to compare against
  currentTime: number;
}

export default function ComparisonView({ prediction, groundTruth, currentTime }: ComparisonViewProps) {
  const getErrorLevel = (pred: WordAlignment, gt: WordAlignment) => {
    const error = Math.abs(pred.start - gt.start);
    if (error < 0.1) return 'bg-green-500';
    if (error < 0.3) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Statistics Card */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 gradient-text">Model Metadata</h3>
          <div className="space-y-3 font-mono text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Model Type:</span>
              <span className="uppercase">{prediction.model}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Processing Time:</span>
              <span>{prediction.metadata.processing_time || '-'}s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Words:</span>
              <span>{prediction.words.length}</span>
            </div>
            {prediction.metadata.rtf && (
              <div className="flex justify-between">
                <span className="text-gray-400">Real-Time Factor:</span>
                <span>{prediction.metadata.rtf}</span>
              </div>
            )}
          </div>
        </div>

        {/* Current Word Info */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4 gradient-text">Current Word</h3>
          {prediction.words.map((word, idx) => {
            if (currentTime >= word.start && currentTime <= word.end) {
              return (
                <div key={idx} className="space-y-2 text-center">
                  <div className="text-4xl font-bold mb-2">{word.text}</div>
                  <div className="font-mono text-sm text-gray-400">
                    {word.start.toFixed(3)}s - {word.end.toFixed(3)}s
                  </div>
                </div>
              );
            }
            return null;
          })}
        </div>
      </div>

      {/* Timeline Visualization */}
      <div className="card overflow-x-auto">
        <h3 className="text-lg font-semibold mb-6 gradient-text">Alignment Timeline</h3>
        <div className="min-w-[800px] space-y-4">
          {/* Timeline Header (Time Scale usually would go here) */}
          
          {/* Prediction Row */}
          <div className="relative h-16 bg-white/5 rounded-lg overflow-hidden">
             {prediction.words.map((word, idx) => (
                <div
                  key={idx}
                  className="absolute top-2 bottom-2 rounded bg-blue-500/30 border border-blue-500/50 flex items-center justify-center text-xs overflow-hidden px-1 truncate hover:z-10 hover:bg-blue-500 hover:text-white transition-all"
                  style={{
                    left: `${(word.start / (prediction.words[prediction.words.length-1]?.end || 100)) * 100}%`,
                    width: `${((word.end - word.start) / (prediction.words[prediction.words.length-1]?.end || 100)) * 100}%`
                  }}
                  title={`${word.text} (${word.start}s - ${word.end}s)`}
                >
                  {word.text}
                </div>
             ))}
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 text-xs text-gray-400">
             <div className="flex items-center gap-2">
               <div className="w-3 h-3 bg-blue-500/50 rounded border border-blue-500/50"></div>
               Prediction
             </div>
             {groundTruth && (
               <div className="flex items-center gap-2">
                 <div className="w-3 h-3 bg-green-500/50 rounded border border-green-500/50"></div>
                 Ground Truth
               </div>
             )}
          </div>
        </div>
      </div>
    </div>
  );
}
