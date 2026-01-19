import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause, Volume2, VolumeX, FastForward, Rewind } from 'lucide-react';
import { cn, formatTime } from '@/lib/utils';
import Image from 'next/image';

interface AudioPlayerProps {
  url: string;
  onTimeUpdate?: (time: number) => void;
  onReady?: () => void;
}

export interface AudioPlayerRef {
  seekTo: (time: number) => void;
  play: () => void;
  pause: () => void;
}

const AudioPlayer = forwardRef<AudioPlayerRef, AudioPlayerProps>(({ url, onTimeUpdate, onReady }, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useImperativeHandle(ref, () => ({
    seekTo: (time: number) => {
      if (wavesurfer.current) {
        wavesurfer.current.setTime(time);
      }
    },
    play: () => {
      wavesurfer.current?.play();
      setIsPlaying(true);
    },
    pause: () => {
      wavesurfer.current?.pause();
      setIsPlaying(false);
    }
  }));

  const onReadyRef = useRef(onReady);
  const onTimeUpdateRef = useRef(onTimeUpdate);

  useEffect(() => {
    onReadyRef.current = onReady;
    onTimeUpdateRef.current = onTimeUpdate;
  }, [onReady, onTimeUpdate]);

  useEffect(() => {
    if (!containerRef.current) return;

    wavesurfer.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#38bdf8',
      progressColor: '#d946ef',
      cursorColor: '#ffffff',
      barWidth: 2,
      barGap: 3,
      barRadius: 3,
      height: 128,
      normalize: true,
      minPxPerSec: 100,
    });

    wavesurfer.current.on('ready', () => {
      setDuration(wavesurfer.current?.getDuration() || 0);
      onReadyRef.current?.();
    });

    wavesurfer.current.on('audioprocess', (time) => {
      setCurrentTime(time);
      onTimeUpdateRef.current?.(time);
    });

    wavesurfer.current.on('interaction', () => {
      const time = wavesurfer.current?.getCurrentTime() || 0;
      setCurrentTime(time);
      onTimeUpdateRef.current?.(time);
    });
    
    wavesurfer.current.on('finish', () => {
      setIsPlaying(false);
    });

    return () => {
      wavesurfer.current?.destroy();
    };
  }, []);

  useEffect(() => {
    if (url && wavesurfer.current) {
      wavesurfer.current.load(url);
    }
  }, [url]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (wavesurfer.current) {
      const newMuted = !isMuted;
      wavesurfer.current.setVolume(newMuted ? 0 : volume);
      setIsMuted(newMuted);
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
    wavesurfer.current?.setVolume(newVolume);
  };

  const skip = (seconds: number) => {
    if (wavesurfer.current) {
      wavesurfer.current.skip(seconds);
    }
  };

  return (
    <div className="w-full space-y-4">
      <div className="waveform-container glass" ref={containerRef} />
      
      <div className="flex items-center justify-between p-4 glass rounded-xl">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => skip(-5)}
            className="p-2 hover:bg-white/10 rounded-full transition-colors"
          >
            <Rewind className="w-5 h-5" />
          </button>
          
          <button 
            onClick={togglePlay}
            className="p-4 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full hover:scale-105 transition-transform shadow-lg shadow-primary-500/25"
          >
            {isPlaying ? (
              <Pause className="w-6 h-6 text-white" fill="currentColor" />
            ) : (
              <Play className="w-6 h-6 text-white" fill="currentColor" />
            )}
          </button>
          
          <button 
            onClick={() => skip(5)}
            className="p-2 hover:bg-white/10 rounded-full transition-colors"
          >
            <FastForward className="w-5 h-5" />
          </button>
          
          <div className="text-sm font-mono text-gray-300 ml-2">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button onClick={toggleMute} className="p-2 hover:bg-white/10 rounded-full">
            {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={isMuted ? 0 : volume}
            onChange={handleVolumeChange}
            className="w-24 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:rounded-full"
          />
        </div>
      </div>
    </div>
  );
});

AudioPlayer.displayName = 'AudioPlayer';

export default AudioPlayer;
