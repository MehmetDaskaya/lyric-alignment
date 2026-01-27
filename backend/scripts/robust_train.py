
import subprocess
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import glob
import logging

# Configuration
# Turkey is UTC+3. 
# We want to stop at 8 AM Turkey time.
# Current time is UTC+3.
TARGET_HOUR = 8 
TARGET_MINUTE = 0

# Paths
BASE_DIR = Path("/Users/mehmetdaskaya/Documents/projects/ai-lyric-alignment")
TRAIN_SCRIPT = BASE_DIR / "backend/training/train.py"
CHECKPOINT_DIR = BASE_DIR / "backend/outputs/checkpoints"
MANIFEST_PATH = BASE_DIR / "backend/data/training_sets/dali_segmented_train.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / "backend/outputs/robust_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RobustTrainer")

def get_latest_checkpoint():
    """Finds the latest .ckpt file in the checkpoint directory."""
    if not CHECKPOINT_DIR.exists():
        return None
    
    # Prioritize 'last.ckpt' if it exists, as it's the specific resume point
    last_ckpt = CHECKPOINT_DIR / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)
        
    # Otherwise check regular checkpoints
    checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))
    if not checkpoints:
        return None
        
    # Sort by modification time
    latest = max(checkpoints, key=os.path.getmtime)
    return str(latest)

def is_past_deadline():
    # Get current time in local timezone (which is TRT according to prompt)
    now = datetime.now()
    if now.hour >= TARGET_HOUR and now.minute >= TARGET_MINUTE:
        # Check if it's actually the next day matching 8AM or if we are already past it today?
        # Assuming we start at 1AM, 8AM is later today.
        # If we start at 9AM, 8AM is tomorrow.
        # Simple check: If hour is >= 8, stop.
        # But if we start at 11 PM, we want to run until 8 AM next day.
        # Given prompt context: "keep the model training through the night until 8am"
        # Since it's 1AM now, stopping at 8AM means simply checking hour == 8.
        return True
    return False

def run_training_loop():
    logger.info("Starting robust training loop...")
    
    # Initial batch size
    batch_size = 2
    
    while True:
        if is_past_deadline():
            logger.info("Reached deadline (8 AM). Stopping training.")
            break
            
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--manifest", str(MANIFEST_PATH),
            "--epochs", "20", # Increase epochs to ensure we run long enough
            "--batch_size", str(batch_size),
            "--gpus", "1"
        ]
        
        # Check for resume
        ckpt = get_latest_checkpoint()
        if ckpt:
            logger.info(f"Resuming from checkpoint: {ckpt}")
            cmd.extend(["--resume_from", ckpt])
        else:
            logger.info("Starting from scratch (no checkpoints found).")
            
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        logger.info(f"Launching command: {' '.join(cmd)}")
        
        try:
            # Run the command
            process = subprocess.Popen(
                cmd, 
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR),
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                sys.stdout.write(line) # Print to main log
                
            process.wait()
            
            if process.returncode == 0:
                logger.info("Training finished successfully!")
                break
            else:
                logger.error(f"Training failed with exit code {process.returncode}")
                # Analyze failure
                if process.returncode == 139: # Segmentation fault / OOM often
                    logger.warning("Detected possible OOM/Segfault.")
                    if batch_size > 1:
                        logger.info(f"Reducing batch size from {batch_size} to {batch_size // 2}")
                        batch_size = batch_size // 2
                    else:
                        logger.error("Batch size is already 1. Cannot reduce further. Waiting 60s and retrying...")
                
                logger.info("Restarting in 30 seconds...")
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("Stopping supervisor...")
            break
        except Exception as e:
            logger.error(f"Supervisor unexpected error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_training_loop()
