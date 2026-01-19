'use client';

import { useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';
import { WordAlignment } from '@/lib/api';

interface LyricsDisplayProps {
  words: WordAlignment[];
  currentTime: number;
  onWordClick?: (time: number) => void;
}

export default function LyricsDisplay({ words, currentTime, onWordClick }: LyricsDisplayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    // Auto scroll to active word
    if (activeRef.current && containerRef.current) {
      const container = containerRef.current;
      const element = activeRef.current;
      
      const elementTop = element.offsetTop;
      const elementHeight = element.offsetHeight;
      const containerHeight = container.offsetHeight;
      
      const scrollPosition = elementTop - containerHeight / 2 + elementHeight / 2;
      
      container.scrollTo({
        top: Math.max(0, scrollPosition),
        behavior: 'smooth'
      });
    }
  }, [currentTime]);

  return (
    <div 
      ref={containerRef}
      className="karaoke-container h-[400px] overflow-y-auto p-6 glass rounded-2xl custom-scrollbar text-center justify-center content-start"
    >
      {words.map((word, index) => {
        const isActive = currentTime >= word.start && currentTime <= word.end;
        const isPast = currentTime > word.end;
        
        return (
          <span
            key={`${index}-${word.start}`}
            ref={isActive ? activeRef : null}
            onClick={() => onWordClick?.(word.start)}
            className={cn(
              "word-highlight cursor-pointer px-1 rounded transition-all duration-300",
              isActive && "active text-2xl font-bold text-white",
              isPast && "text-gray-400 opacity-60",
              !isActive && !isPast && "text-gray-300 opacity-80 hover:text-white"
            )}
            title={`Start: ${word.start.toFixed(2)}s, End: ${word.end.toFixed(2)}s`}
          >
            {word.text}
          </span>
        );
      })}
      
      {words.length === 0 && (
        <div className="w-full h-full flex items-center justify-center text-gray-400">
          <p>No lyrics aligned yet. Upload audio and lyrics to start.</p>
        </div>
      )}
    </div>
  );
}
