'use client';

import React, { useState, useEffect } from 'react';
import { Play, Square, CheckCircle, XCircle, Loader2, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Song {
  id: string;
  name: string;
  filename: string;
}

interface BatchJob {
  id: string;
  status: string;
  progress: number;
  total: number;
  current_song: string;
  current_model: string;
  error?: string;
  results?: SongResult[];
}

interface SongResult {
  song_id: string;
  song_name: string;
  status: string;
  processing_time: number;
  metrics: Record<string, ModelMetrics>;
}

interface ModelMetrics {
  processing_time: number;
  rtf: number;
  word_count: number;
}

const API_BASE = '/api';

async function fetchSongs(): Promise<Song[]> {
  const res = await fetch(`${API_BASE}/data/songs`);
  if (!res.ok) throw new Error('Failed to fetch songs');
  return res.json();
}

async function startBatchJob(songIds: string[], models: string[], useSourceSeparation: boolean) {
  const res = await fetch(`${API_BASE}/batch/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      song_ids: songIds,
      models,
      use_source_separation: useSourceSeparation
    })
  });
  if (!res.ok) throw new Error('Failed to start batch job');
  return res.json();
}

async function getJobStatus(jobId: string): Promise<BatchJob> {
  const res = await fetch(`${API_BASE}/batch/${jobId}`);
  if (!res.ok) throw new Error('Failed to get job status');
  return res.json();
}

export default function BatchProcessingPanel() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [selectedSongs, setSelectedSongs] = useState<Set<string>>(new Set());
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set(['dtw', 'hmm', 'ctc']));
  const [useSourceSeparation, setUseSourceSeparation] = useState(false);
  const [currentJob, setCurrentJob] = useState<BatchJob | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    fetchSongs().then(setSongs).catch(console.error);
  }, []);

  // Timer for elapsed time
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (currentJob && currentJob.status === 'running') {
      interval = setInterval(() => {
        setElapsedTime(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [currentJob?.status]);

  // Polling for job status
  useEffect(() => {
    let pollInterval: NodeJS.Timeout;
    
    if (currentJob && currentJob.status === 'running') {
      pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(currentJob.id);
          setCurrentJob(status);
          
          if (status.status === 'completed' || status.status === 'failed') {
            setIsLoading(false);
          }
        } catch (e) {
          console.error('Polling error:', e);
        }
      }, 2000);
    }
    
    return () => clearInterval(pollInterval);
  }, [currentJob?.id, currentJob?.status]);

  const toggleSong = (songId: string) => {
    setSelectedSongs(prev => {
      const next = new Set(prev);
      if (next.has(songId)) next.delete(songId);
      else next.add(songId);
      return next;
    });
  };

  const toggleModel = (model: string) => {
    setSelectedModels(prev => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
  };

  const selectAll = () => {
    setSelectedSongs(new Set(songs.map(s => s.id)));
  };

  const clearSelection = () => {
    setSelectedSongs(new Set());
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStartBatch = async () => {
    if (selectedSongs.size === 0 || selectedModels.size === 0) {
      setError('Please select at least one song and one model');
      return;
    }

    setIsLoading(true);
    setError(null);
    setElapsedTime(0);

    try {
      const response = await startBatchJob(
        Array.from(selectedSongs),
        Array.from(selectedModels),
        useSourceSeparation
      );
      
      setCurrentJob({
        id: response.id,
        status: 'running',
        progress: 0,
        total: selectedSongs.size * selectedModels.size,
        current_song: '',
        current_model: ''
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start batch job');
      setIsLoading(false);
    }
  };

  const progressPercent = currentJob ? Math.round((currentJob.progress / currentJob.total) * 100) : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Batch Processing</h2>
          <p className="text-gray-400 text-sm">Process multiple songs with different models</p>
        </div>
        {currentJob?.status === 'completed' && (
          <div className="flex items-center gap-2 text-green-400">
            <CheckCircle className="w-5 h-5" />
            <span>Completed</span>
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl flex items-center gap-2">
          <XCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {/* Song Selection */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Select Songs</h3>
          <div className="flex gap-2">
            <button onClick={selectAll} className="text-sm text-blue-400 hover:text-blue-300">
              Select All
            </button>
            <span className="text-gray-600">|</span>
            <button onClick={clearSelection} className="text-sm text-gray-400 hover:text-gray-300">
              Clear
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-60 overflow-y-auto">
          {songs.map(song => (
            <label
              key={song.id}
              className={cn(
                "flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all",
                selectedSongs.has(song.id)
                  ? "bg-blue-500/20 border border-blue-500/50"
                  : "bg-white/5 border border-white/10 hover:border-white/20"
              )}
            >
              <input
                type="checkbox"
                checked={selectedSongs.has(song.id)}
                onChange={() => toggleSong(song.id)}
                className="rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-200 truncate">{song.name}</span>
            </label>
          ))}
        </div>
        
        <p className="text-sm text-gray-500 mt-3">
          {selectedSongs.size} of {songs.length} songs selected
        </p>
      </div>

      {/* Model Selection */}
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Select Models</h3>
        
        <div className="flex flex-wrap gap-3">
          {[
            { id: 'dtw', label: 'DTW Baseline', color: 'blue' },
            { id: 'hmm', label: 'HMM Forced Alignment', color: 'green' },
            { id: 'ctc', label: 'Deep Learning (CTC)', color: 'purple' }
          ].map(model => (
            <label
              key={model.id}
              className={cn(
                "flex items-center gap-2 px-4 py-2 rounded-lg cursor-pointer transition-all",
                selectedModels.has(model.id)
                  ? `bg-${model.color}-500/20 border border-${model.color}-500/50 text-${model.color}-400`
                  : "bg-white/5 border border-white/10 text-gray-400 hover:border-white/20"
              )}
            >
              <input
                type="checkbox"
                checked={selectedModels.has(model.id)}
                onChange={() => toggleModel(model.id)}
                className="rounded border-gray-600 bg-gray-800"
              />
              <span className="text-sm">{model.label}</span>
            </label>
          ))}
        </div>

        <label className="flex items-center gap-2 mt-4 text-sm text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={useSourceSeparation}
            onChange={(e) => setUseSourceSeparation(e.target.checked)}
            className="rounded border-gray-600 bg-gray-800 text-blue-500"
          />
          Use Vocal Separation (slower but more accurate)
        </label>
      </div>

      {/* Progress Section */}
      {currentJob && currentJob.status === 'running' && (
        <div className="card border-blue-500/30">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
              <span className="text-white font-medium">Processing...</span>
            </div>
            <span className="text-blue-400 font-mono">{formatTime(elapsedTime)}</span>
          </div>
          
          <div className="w-full bg-gray-800 rounded-full h-3 mb-3">
            <div 
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          
          <div className="flex justify-between text-sm text-gray-400">
            <span>{currentJob.progress} / {currentJob.total} tasks</span>
            <span>{progressPercent}%</span>
          </div>
          
          {currentJob.current_song && (
            <p className="text-sm text-gray-500 mt-2">
              Current: <span className="text-gray-300">{currentJob.current_song}</span>
              {currentJob.current_model && (
                <span className="text-gray-500"> ({currentJob.current_model.toUpperCase()})</span>
              )}
            </p>
          )}
        </div>
      )}

      {/* Results Section */}
      {currentJob?.status === 'completed' && currentJob.results && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-green-400" />
            <h3 className="text-lg font-semibold text-white">Results</h3>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 px-3 text-gray-400">Song</th>
                  <th className="text-left py-2 px-3 text-gray-400">Status</th>
                  <th className="text-right py-2 px-3 text-gray-400">Time (s)</th>
                  <th className="text-right py-2 px-3 text-blue-400">DTW</th>
                  <th className="text-right py-2 px-3 text-green-400">HMM</th>
                  <th className="text-right py-2 px-3 text-purple-400">CTC</th>
                </tr>
              </thead>
              <tbody>
                {currentJob.results.map(result => (
                  <tr key={result.song_id} className="border-b border-white/5 hover:bg-white/5">
                    <td className="py-2 px-3 text-gray-200 truncate max-w-[200px]">{result.song_name}</td>
                    <td className="py-2 px-3">
                      {result.status === 'completed' ? (
                        <span className="text-green-400">✓</span>
                      ) : (
                        <span className="text-red-400">✗</span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-400">
                      {result.processing_time.toFixed(1)}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-300">
                      {result.metrics?.dtw?.word_count || '-'}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-300">
                      {result.metrics?.hmm?.word_count || '-'}
                    </td>
                    <td className="py-2 px-3 text-right text-gray-300">
                      {result.metrics?.ctc?.word_count || '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Start Button */}
      <button
        onClick={handleStartBatch}
        disabled={isLoading || selectedSongs.size === 0 || selectedModels.size === 0}
        className={cn(
          "w-full py-4 rounded-xl font-semibold text-lg transition-all flex items-center justify-center gap-2",
          isLoading
            ? "bg-gray-700 text-gray-400 cursor-not-allowed"
            : "bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700"
        )}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <Play className="w-5 h-5" />
            Start Batch Processing ({selectedSongs.size} songs × {selectedModels.size} models)
          </>
        )}
      </button>
    </div>
  );
}
