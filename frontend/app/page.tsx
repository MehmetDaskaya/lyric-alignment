'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import AudioPlayer, { AudioPlayerRef } from '@/components/AudioPlayer';
import LyricsDisplay from '@/components/LyricsDisplay';
import ComparisonView from '@/components/ComparisonView';
import { api, AlignmentResult } from '@/lib/api';
import { AlignJustify, Upload, Mic2, Activity, BarChart, PlayCircle, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import LoadingModal from '@/components/LoadingModal';

export default function Home() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("Initializing...");
  const [results, setResults] = useState<AlignmentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'karaoke' | 'comparison'>('karaoke');
  
  const playerRef = useRef<AudioPlayerRef>(null);

  /* Song Selection Logic */
  const [songs, setSongs] = useState<{ id: string; name: string }[]>([]);
  const [selectedSongId, setSelectedSongId] = useState<string>('');
  
  // Settings
  const [selectedModel, setSelectedModel] = useState<string>('dtw');
  const [useSourceSeparation, setUseSourceSeparation] = useState<boolean>(false);

  useEffect(() => {
    // Fetch available songs
    api.getSongs().then(setSongs).catch(console.error);
  }, []);

  const handleInitialize = async () => {
    if (!selectedSongId) return;
    setIsProcessing(true);
    setProcessingStatus("Loading DALI data...");
    try {
      const { task_id } = await api.processDaliSong(selectedSongId);
      setTaskId(task_id);
      setResults(null); // Clear previous results
      setIsProcessing(false);
    } catch (err) {
      console.error('Initialization failed', err);
      setError("Failed to initialize song");
      setIsProcessing(false);
    }
  };

  const handleAlign = async (modelOverride?: string) => {
    if (!taskId) return;
    
    // Allow button click to override selected model
    const modelToUse = modelOverride || selectedModel;
    if (modelOverride) setSelectedModel(modelOverride);

    setIsProcessing(true);
    setError(null);
    setProcessingStatus("Requesting alignment...");
    
    try {
      // 1. Start the process
      await api.processAlignment(taskId, modelToUse, useSourceSeparation);
      
      // 2. Poll for status
      const pollingInterval = setInterval(async () => {
        try {
          const status = await api.getTaskStatus(taskId);
          
          setProcessingStatus(status.progress || "Processing...");
          
          if (status.status === 'completed') {
            clearInterval(pollingInterval);
            
            // 3. Fetch results
            setProcessingStatus("Fetching results...");
            const result = await api.getResults(taskId, modelToUse);
            setResults(result);
            setIsProcessing(false);
          } else if (status.status === 'error') {
            clearInterval(pollingInterval);
            setError(status.error || "Alignment failed");
            setIsProcessing(false);
          }
        } catch (e) {
          console.error("Polling error:", e);
           // Don't stop polling on transient errors
        }
      }, 1000); // Poll every second
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Alignment failed');
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen p-8 max-w-7xl mx-auto space-y-8 pb-32">
      <LoadingModal 
        isOpen={isProcessing} 
        model={selectedModel} 
        progress={processingStatus} 
      />

      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
            Lyric-Audio Alignment
          </h1>
          <p className="text-gray-400 max-w-xl">
            Select a song from the dataset to compare alignment models.
          </p>
        </div>
        <Link 
          href="/batch"
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/50 rounded-xl text-purple-400 hover:text-purple-300 hover:border-purple-400 transition-all"
        >
          <Layers className="w-5 h-5" />
          Batch Processing
        </Link>
      </header>
      
      {error && (
        <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl">
            {error}
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Sidebar - Controls */}
        <div className="lg:col-span-4 space-y-6">
          <div className="card space-y-6">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Upload className="w-5 h-5 text-blue-400" />
              Select DALI Song
            </h2>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm text-gray-400">Available Songs</label>
                <select 
                  className="w-full bg-black/20 border border-white/10 rounded-xl p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none transition-all cursor-pointer"
                  value={selectedSongId}
                  onChange={(e) => setSelectedSongId(e.target.value)}
                >
                  <option value="">Select a song...</option>
                  {songs.map(song => (
                    <option key={song.id} value={song.id}>
                      {song.name}
                    </option>
                  ))}
                </select>
              </div>

              <button 
                onClick={handleInitialize}
                disabled={isProcessing || !selectedSongId}
                className={cn(
                  "w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed",
                  taskId && "bg-green-500 hover:bg-green-600"
                )}
              >
                {taskId ? 'Session Ready (Reset)' : 'Initialize Session'}
              </button>
            </div>
          </div>

          {/* Model Selection */}
          <div className={cn("card space-y-4 transition-all duration-500", !taskId && "opacity-50 pointer-events-none blur-sm")}>
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-400" />
                Run Alignment
                </h2>
                
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                    <input 
                        type="checkbox"
                        checked={useSourceSeparation}
                        onChange={(e) => setUseSourceSeparation(e.target.checked)}
                        className="rounded border-white/20 bg-white/5 text-blue-500 focus:ring-blue-500"
                    />
                    Use Vocal Separation
                </label>
            </div>
            
            <div className="grid gap-3">
              <button 
                onClick={() => handleAlign('dtw')}
                disabled={isProcessing}
                className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-blue-500/50 transition-all group disabled:opacity-50"
              >
                <div className="text-left">
                  <div className="font-semibold text-blue-400 group-hover:text-blue-300">DTW Baseline</div>
                  <div className="text-xs text-gray-500">Dynamic Time Warping</div>
                </div>
                <PlayCircle className="w-6 h-6 text-gray-600 group-hover:text-blue-500 transition-colors" />
              </button>

              <button 
                onClick={() => handleAlign('hmm')}
                disabled={isProcessing}
                className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-green-500/50 transition-all group disabled:opacity-50"
              >
                <div className="text-left">
                  <div className="font-semibold text-green-400 group-hover:text-green-300">HMM Forced Alignment</div>
                  <div className="text-xs text-gray-500">Probabilistic Model</div>
                </div>
                <PlayCircle className="w-6 h-6 text-gray-600 group-hover:text-green-500 transition-colors" />
              </button>

              <button 
                onClick={() => handleAlign('ctc')}
                disabled={isProcessing}
                className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 hover:border-purple-500/50 transition-all group disabled:opacity-50"
              >
                <div className="text-left">
                  <div className="font-semibold text-purple-400 group-hover:text-purple-300">Deep Learning (CTC)</div>
                  <div className="text-xs text-gray-500">wav2vec2 + CTC</div>
                </div>
                <PlayCircle className="w-6 h-6 text-gray-600 group-hover:text-purple-500 transition-colors" />
              </button>
            </div>
          </div>
        </div>

        {/* Right Content - Visualization */}
        <div className="lg:col-span-8 space-y-6">
          {/* Audio Player */}
          <div className={cn("transition-all duration-500", !taskId && "opacity-50 pointer-events-none")}>
            <AudioPlayer 
              ref={playerRef}
              url={taskId ? api.getAudioUrl(taskId) : ''} 
              onTimeUpdate={setCurrentTime}
            />
          </div>

          {/* Visualization Tabs */}
          <div className={cn("space-y-6 transition-all duration-500", !results && "opacity-50 pointer-events-none blur-sm")}>
            <div className="flex gap-4 border-b border-white/10 pb-4">
              <button 
                onClick={() => setActiveTab('karaoke')}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg transition-all",
                  activeTab === 'karaoke' ? "bg-white/10 text-white" : "text-gray-500 hover:text-gray-300"
                )}
              >
                <Mic2 className="w-4 h-4" />
                Karaoke View
              </button>
              <button 
                onClick={() => setActiveTab('comparison')}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg transition-all",
                  activeTab === 'comparison' ? "bg-white/10 text-white" : "text-gray-500 hover:text-gray-300"
                )}
              >
                <BarChart className="w-4 h-4" />
                Comparison & Metrics
              </button>
            </div>

            <div className="min-h-[400px]">
              {selectedModel && activeTab === 'karaoke' ? (
                <LyricsDisplay 
                  words={results?.words || []} 
                  currentTime={currentTime}
                  onWordClick={(time) => {
                    playerRef.current?.seekTo(time);
                  }} 
                />
              ) : (
                results && <ComparisonView prediction={results} currentTime={currentTime} />
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
