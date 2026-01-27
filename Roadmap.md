## Phase 1: Data Integration & Cleaning (Done)
- [x] **DALI v1.0 Integration**: Configured system to read annotations directly from `DALI_v1.0` in project root.
- [x] **Unified Data Service**: Created `DaliService` to handle data loading, replacing scattered scripts.
- [x] **Audio Acquisition**: 
    - Installed `ffmpeg`.
    - Created `backend/scripts/download_dataset_audio.py` for bulk downloading.
    - Verified download on dev set.

## Phase 2: Pipeline Robustness & Baselines (Done)
- [x] **Verification**: `verify_batch.py` runs DTW baseline on local data.
- [x] **Metric Validation**: Validated flow of Ground Truth loading.

## Phase 3: Custom Model Training (In Progress)
To obtain state-of-the-art results, we will move beyond baseline algorithms (DTW/HMM) and pre-trained generl models (wav2vec2-base) to a specialized architecture.

### 3.1 Data Preparation (Done)
- [x] **Filtering**: Created `backend/scripts/prepare_training_data.py` to generate `dali_train_manifest.jsonl`.
- [x] **Preprocessing**: Implemented `DALIDataset` and `DataCollatorCTCWithPadding` in `backend/training/dataset.py`.

### 3.2 Model Architecture Design (Done)
- [x] **Architecture**: Implemented `LyricAlignmentModel` wrapping `Wav2Vec2ForCTC` in `backend/training/model.py`.
- [x] **Training Script**: Created `backend/training/train.py` using PyTorch Lightning.

### 3.3 Training Pipeline (Verified)
- [x] **Framework Setup**: Installed PyTorch Lightning and dependencies.
- [x] **Dry Run**: Verified training loop execution on Apple Silicon (MPS).
- [x] **Audio Loading Fix**: Switched to `librosa` to resolve `TorchCodec` / `soundfile` warnings for MP3 loading.
- [/] **Full Training**: Running stabilized job (64 songs, 20 epochs, LR=5e-5). Logs: `backend/outputs/training_history.jsonl`.

### 3.4 Deployment
- Export trained model to ONNX for fast inference.
- Update `AlignmentService` to load this custom model checkpoint.

## Immediate Next Steps
1.  **Install FFmpeg**: `brew install ffmpeg` (on macOS).
2.  **Verify Audio**: Run `python backend/scripts/prepare_dali.py` until successful.
3.  **Run Benchmarks**: Execute batch processing on 10 DALI songs to establish a baseline MAE for existing models.
