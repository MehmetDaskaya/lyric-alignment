"""
Training Script for Lyric Alignment Model
"""
import logging
import argparse
from pathlib import Path
import sys
import os
import json
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import Wav2Vec2Processor

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset import DALIDataset, DataCollatorCTCWithPadding
from training.model import LyricAlignmentModel
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)  # Low default for safety
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpus", type=int, default=0 if not torch.cuda.is_available() and not torch.backends.mps.is_available() else 1)
    parser.add_argument("--manifest", type=str, default=str(settings.DATA_DIR / "training_sets" / "dali_train_manifest.jsonl"))
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    logger.info(f"Starting training with args: {args}")

    # 1. Load Processor
    processor_name = "facebook/wav2vec2-base-960h"
    logger.info(f"Loading processor: {processor_name}")
    processor = Wav2Vec2Processor.from_pretrained(processor_name)

    # 2. Data Module
    logger.info("Loading dataset...")
    collator = DataCollatorCTCWithPadding(processor=processor)
    
    full_dataset = DALIDataset(
        manifest_path=args.manifest,
        processor=processor,
        max_duration=30.0 # Limit duration to avoid OOM
    )
    
    if len(full_dataset) == 0:
        logger.error("Dataset is empty. Run prepare_training_data.py first.")
        return

    # Split Train/Val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collator,
        num_workers=0
    )

    # 3. Model
    logger.info("Initializing model...")
    model = LyricAlignmentModel(
        model_name=processor_name,
        learning_rate=args.lr,
        processor=processor
    )

    # 4. Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.OUTPUT_DIR / "checkpoints",
        filename="lyric-align-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_last=True # Explicitly save the last state for resuming
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    
    # Check for MPS (Metal Performance Shaders) on Mac
    accelerator = "auto"
    devices = 1
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        logger.info("Using MPS (Apple Silicon) acceleration")
    elif torch.cuda.is_available():
        accelerator = "gpu"
        
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=1
    )

    # 5. Train
    logger.info("Starting training loop...")
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)
    
    # 6. Save History
    logger.info("Training complete. Saving run history...")
    history_file = settings.OUTPUT_DIR / "training_history.jsonl"
    
    # Get best validation loss if available
    best_loss = trainer.callback_metrics.get("val_loss")
    if best_loss:
        best_loss = float(best_loss)
    
    run_record = {
        "timestamp": str(datetime.now()),
        "model_name": processor_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "final_val_loss": best_loss,
        "args": vars(args),
        "comments": "User requested training on larger dataset"
    }
    
    with open(history_file, 'a') as f:
        f.write(json.dumps(run_record) + "\n")
        
    logger.info(f"Run recorded in {history_file}")

if __name__ == "__main__":
    main()
