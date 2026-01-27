# Lyric Alignment Project - Developer Handover Guide

## Project Overview
This project aims to build an AI model that automatically aligns lyrics with audio timestamps. We have moved from simple algorithms (DTW, HMM) to a **Deep Learning approach** using a fine-tuned Wav2Vec2 CTC model trained on the **DALI v1.0 dataset**.

## 1. System Architecture

### **Dataset Integration (DALI)**
*   **Source**: The `DALI_v1.0` folder in the project root contains the raw dataset (audio URLs and ground-truth text/time annotations in `.gz` files).
*   **Service Layer**: `backend/services/dali_service.py` is the central access point. It handles:
    *   Reading `.gz` annotations.
    *   Downloading audio from YouTube using `yt-dlp`.
    *   Serving paths and metadata to the rest of the app.
*   **Storage**: Downloaded audio is stored in `backend/data/dali_audio/` (or `data/dali_audio/`).

### **Training Pipeline (`backend/training/`)**
1.  **Preparation**:
    *   `backend/scripts/prepare_segmented_manifest.py`: This is the **critical** first step. It reads DALI annotations and splits songs into small, aligned chunks (~15 seconds). It saves this list to `backend/data/training_sets/dali_segmented_train.jsonl`.
    *   *Why segmentation?* Training on full songs (3 mins) causes memory issues and model "collapse" (predicting nothing). Short segments fix this.

2.  **Model**:
    *   `LyricAlignmentModel` (`backend/training/model.py`): A PyTorch Lightning wrapper around `facebook/wav2vec2-base-960h`.
    *   It uses **CTC Loss** to learn the alignment between audio waveforms and text characters.

3.  **Data Loading**:
    *   `DALIDataset` (`backend/training/dataset.py`): Loads specific audio slices defined in the manifest.
    *   **Note**: It uses `librosa` as a fallback for loading MP3s because `torchaudio` has issues with some codecs on macOS.

4.  **Execution**:
    *   `train.py`: The standard training script.
    *   `backend/scripts/robust_train.py`: A **supervisor script** created to run training overnight. It auto-restarts the training if it crashes (e.g., Out of Memory) and resumes from the last checkpoint.

## 2. Current Status (IMPORTANT)

**Last Run**: Overnight session on Jan 27, 2026.
*   **Progress**: Epoch 0 completed successfully. Epoch 1 reached ~15%.
*   **Checkpoints**: Saved in `backend/outputs/checkpoints/` (e.g., `lyric-align-epoch=00...`).
*   **Issue**: Training is **very slow** (~1.5 minutes per batch on M1/M2 Mac).
    *   *Reason*: We are loading and resampling MP3s on the fly with a single CPU thread (`num_workers=0`).

## 3. Recommended Next Steps

### **A. Optimization (Immediate Priority)**
The training is too slow to specific usable results quickly.
1.  **Pre-process Audio**: Instead of resampling MP3s every time, write a script to convert all DALI audio to **16kHz Mono WAV** files once. Save them to a new folder.
2.  **Update Dataset**: Point `DALIDataset` to these pre-processed WAVs.
3.  **Parallel Loading**: Set `num_workers=4` in `train.py`. This is currently 0 because Python multiprocessing + MacOS MPS + Librosa is unstable. Pre-processing audio removes the Librosa dependency during training, allowing safe parallelism.

### **B. Continue Training**
Once optimized:
1.  Run `python backend/training/train.py --manifest ... --resume_from backend/outputs/checkpoints/last.ckpt`.
2.  Aim for ~20 epochs.

### **C. Evaluation**
1.  Use `verify_batch.py` (updated to use the trained checkpoint) to align a test song.
2.  Compare the predicted timestamps against the DALI ground truth.

## 4. Key Commands

**1. Create Training Data (if needed again)**
```bash
python backend/scripts/prepare_segmented_manifest.py
```

**2. Run Training (Standard)**
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python backend/training/train.py --manifest backend/data/training_sets/dali_segmented_train.jsonl --batch_size 4
```

**3. Run Training (Robust / Overnight)**
```bash
python backend/scripts/robust_train.py
```

**4. Test/Verify**
```bash
python verify_batch.py
```
*(Note: verify_batch.py might need updates to load your specific checkpoint instead of the baseline)*

## 5. File Structure
*   `backend/`: Core logic.
    *   `services/`: Data access (`dali_service.py`).
    *   `training/`: Deep learning code (`model.py`, `dataset.py`, `train.py`).
    *   `scripts/`: Utility scripts (`prepare_...`, `robust_train.py`).
    *   `outputs/`: Logs and Checkpoints.
*   `DALI_v1.0/`: The raw dataset (Ignored by Git).

---
**Note for Friend**: The hard part (pipeline setup, data cleaning, model definition) is done. The model *can* learn. The remaining challenge is just **engineering the data loading** to be fast enough to finish training in a reasonable time.
