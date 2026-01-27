import logging
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pytorch_lightning as pl
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class LyricAlignmentModel(pl.LightningModule):
    """
    Lightning Module for CTC-based Lyric Alignment.
    Wraps Hugging Face Wav2Vec2ForCTC.
    """
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-base-960h",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.005,
        vocab_size: int = None,
        processor: Wav2Vec2Processor = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load Pretrained model
        # Note: If we had a custom vocab, we'd resize the head. 
        # For now, we reuse the pretrained base.
        try:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name, 
                ctc_loss_reduction="mean", 
                pad_token_id=processor.tokenizer.pad_token_id,
                ctc_zero_infinity=True
            )
            # Freeze extraction layers to save memory/time?
            self.model.freeze_feature_extractor()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def forward(self, input_values, attention_mask=None, labels=None):
        return self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_values=batch['input_values'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_values=batch['input_values'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels']
        )
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        
        # Calculate WER/CER here if desired, but expensive
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
