"""
Benchmark Existing Alignment Models (DTW, HMM, CTC)
Runs a batch job on all available local DALI audio files and reports comparative metrics.
"""
import asyncio
import sys
import logging
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from services.batch_processor import batch_processor, JobStatus
from services.dali_service import dali_service
from config import settings

# Configure logging
logging.basicConfig(level=logging.ERROR) # Only show errors to keep output clean
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

async def run_benchmark():
    logger.info("Initializing Model Benchmark...")
    
    # 1. Identify valid local songs
    all_ids = dali_service.get_all_ids()
    valid_ids = []
    
    for entry_id in all_ids:
        # Check if audio is already downloaded (don't download new ones for benchmark speed)
        if dali_service.get_audio_path(entry_id, download_if_missing=False):
            valid_ids.append(entry_id)
            
    if not valid_ids:
        logger.error("No downloaded audio files found. Run backend/scripts/download_dataset_audio.py first.")
        return

    # User requested quick comparison, so limit to 1 song for speed
    valid_ids = valid_ids[:1]
    logger.info(f"Found {len(valid_ids)} songs with local audio. Starting benchmark...")

    # 2. Create Batch Job
    # We want to compare DTW, HMM, and CTC (Wav2Vec2 zero-shot)
    models_to_test = ['dtw', 'hmm', 'ctc']
    
    job = batch_processor.create_job(
        song_ids=valid_ids,
        models=models_to_test,
        use_source_separation=False # Disable for speed, enable for better accuracy if desired
    )
    
    await batch_processor.start_job(job.id)
    
    # 3. Monitor Progress
    print(f"\nProcessing Job ID: {job.id}")
    while job.status == JobStatus.RUNNING:
        sys.stdout.write(f"\rProgress: {job.progress}/{job.total} tasks completed...")
        sys.stdout.flush()
        await asyncio.sleep(0.5)
    
    print("\n\nAnalysis Complete!")
    
    # 4. Aggregate Results
    results = []
    
    if job.status == JobStatus.COMPLETED:
        for song_res in job.results:
            row_base = {
                'Song': song_res.song_name,
                'ID': song_res.song_id[:8]
            }
            
            for model in models_to_test:
                metrics = song_res.metrics.get(model, {})
                if 'error' in metrics:
                    continue
                    
                # Creating a row for each model-song combo or wide format?
                # Let's do wide format for easier comparison per song
                mae = metrics.get('mae', 'N/A')
                acc = metrics.get('accuracy', 'N/A')
                time = metrics.get('processing_time', 'N/A')
                
                results.append({
                    'Song': song_res.song_name[:20],
                    'Model': model.upper(),
                    'MAE (s)': mae,
                    'Accuracy (%)': acc,
                    'Time (s)': time
                })
    
    # 5. Display Table
    if results:
        df = pd.DataFrame(results)
        
        # Summary by Model
        print("\n=== Average Performance by Model ===")
        try:
            summary = df.groupby('Model')[['MAE (s)', 'Accuracy (%)', 'Time (s)']].mean()
            print(tabulate(summary, headers='keys', tablefmt='github', floatfmt=".3f"))
        except:
            print("Could not calculate averages (check for missing data).")

        print("\n=== Detailed Results ===")
        print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
        
    else:
        logger.error("No results produced. Check logs.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
