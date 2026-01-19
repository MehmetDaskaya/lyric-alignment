export interface WordAlignment {
  text: string;
  start: number;
  end: number;
}

export interface AlignmentResult {
  model: string;
  words: WordAlignment[];
  metadata: {
    processing_time?: number;
    rtf?: number;
    dtw_cost?: number;
    n_phonemes?: number;
    n_frames?: number;
    frame_rate?: number;
    source_separation?: boolean;
  };
}

export interface EvaluationMetrics {
  mae: number;
  mae_end: number;
  pc: number;
  tolerance: number;
  matched_words: number;
  total_pred: number;
  total_gt: number;
  error_std: number;
  median_error: number;
  max_error: number;
}

export interface TaskStatus {
  id: string;
  status: string;
  progress: string;
  error?: string;
  results: string[];
}

const API_BASE = '/api';

export const api = {
  async getSongs(): Promise<{ id: string; name: string; filename: string }[]> {
    const res = await fetch(`${API_BASE}/data/songs`);
    if (!res.ok) throw new Error('Failed to fetch songs');
    return res.json();
  },

  async processDaliSong(songId: string): Promise<{ task_id: string; status: string; message: string }> {
    const res = await fetch(`${API_BASE}/process_dali/${songId}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to process DALI song');
    return res.json();
  },

  async processAlignment(
    taskId: string, 
    model: string, 
    useSourceSeparation: boolean = true
  ): Promise<{ task_id: string; status: string; message: string }> {
    const formData = new FormData();
    formData.append('model', model);
    formData.append('use_source_separation', String(useSourceSeparation));
    
    const res = await fetch(`${API_BASE}/process/${taskId}`, {
      method: 'POST',
      body: formData,
    });
    
    if (!res.ok) throw new Error('Alignment processing failed');
    return res.json();
  },

  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const res = await fetch(`${API_BASE}/status/${taskId}`);
    if (!res.ok) throw new Error('Failed to get status');
    return res.json();
  },

  async getResults(taskId: string, model: string): Promise<AlignmentResult> {
    const url = `${API_BASE}/results/${taskId}?model=${model}`;
      
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch results');
    return res.json();
  },

  getAudioUrl(taskId: string, separated: boolean = false): string {
    return `${API_BASE}/audio/${taskId}?separated=${separated}`;
  },

  async evaluate(
    prediction: AlignmentResult, 
    groundTruth: AlignmentResult,
    tolerance: number = 0.3
  ): Promise<EvaluationMetrics> {
    const res = await fetch(`${API_BASE}/evaluate?tolerance=${tolerance}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prediction,
        ground_truth: groundTruth
      }),
    });
    
    if (!res.ok) throw new Error('Evaluation failed');
    return res.json();
  }
};
