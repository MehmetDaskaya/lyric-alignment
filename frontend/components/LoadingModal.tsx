import React, { useState, useEffect, useRef } from 'react';

interface LoadingModalProps {
  isOpen: boolean;
  model: string;
  progress: string;
}

export default function LoadingModal({ isOpen, model, progress }: LoadingModalProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (isOpen) {
      // Reset and start timer when modal opens
      setElapsedSeconds(0);
      intervalRef.current = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      // Clear timer when modal closes
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isOpen]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-8 max-w-md w-full shadow-2xl relative overflow-hidden">
        {/* Animated Background Glow */}
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 animate-pulse"></div>
        
        <div className="flex flex-col items-center text-center space-y-6">
          {/* Spinner with Timer */}
          <div className="relative">
            <div className="w-20 h-20 border-4 border-gray-700 border-t-blue-500 rounded-full animate-spin"></div>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-xs font-bold text-gray-400">{model.toUpperCase()}</span>
              <span className="text-sm font-mono text-blue-400">{formatTime(elapsedSeconds)}</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <h3 className="text-xl font-bold text-white">Aligning Lyrics</h3>
            <p className="text-gray-400 text-sm">Please wait while our AI processes your audio.</p>
          </div>

          {/* Progress Message */}
          <div className="w-full bg-gray-800 rounded-lg p-4 border border-gray-700/50">
            <p className="text-blue-400 font-mono text-sm animate-pulse">
              {progress || "Initializing..."}
            </p>
          </div>
          
          <p className="text-xs text-gray-500">
            This may take 1-2 minutes depending on song length.
          </p>
        </div>
      </div>
    </div>
  );
}
