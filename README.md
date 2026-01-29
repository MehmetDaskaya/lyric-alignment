# AI Lyric Alignment - Project Submission

This project implements and compares three approaches to automatic lyric-audio alignment: Dynamic Time Warping (DTW), Hidden Markov Models (HMM), and deep learning with CTC-based wav2vec2 fine-tuning.

## Project Report

The complete technical report is in: **`project proposal.txt`**

This IEEE-format document contains:
- Detailed methodology for all three approaches
- Implementation architecture and design decisions  
- Training results and analysis
- Honest assessment of challenges and limitations
- Future work recommendations

## Repository Structure

```
├── backend/
│   ├── models/              # DTW, HMM, and CTC alignment implementations
│   ├── services/            # DALI dataset integration
│   ├── training/            # Deep learning training pipeline
│   ├── scripts/             # Data preparation and robust training
│   ├── outputs/             # Training checkpoints and logs
│   └── config.py            # Configuration settings
├── benchmark_models.py      # Benchmarking system for all methods
├── requirements.txt         # Python dependencies
├── project proposal.txt     # **Main technical report**
└── README.md               # This file
```

## Key Implementation Files

### Baseline Methods
- `backend/models/dtw_aligner.py` - DTW with MFCC features
- `backend/models/hmm_aligner.py` - HMM-based Viterbi alignment
- `backend/models/ctc_aligner.py` - CTC forced alignment baseline

### Deep Learning
- `backend/training/model.py` - wav2vec2 fine-tuning wrapper
- `backend/training/dataset.py` - DALI dataset loader with segmentation
- `backend/training/train.py` - Training script
- `backend/scripts/prepare_segmented_manifest.py` - Audio segmentation pipeline

### Infrastructure
- `backend/services/dali_service.py` - DALI dataset management
- `backend/scripts/robust_train.py` - Training supervisor with auto-restart
- `benchmark_models.py` - Multi-method evaluation framework

## Setup Instructions

### Requirements
- Python 3.13
- Apple Silicon Mac (for MPS acceleration) or NVIDIA GPU
- ~50GB storage for DALI dataset
- Dependencies in `requirements.txt`

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download DALI dataset (place DALI_v1.0 folder in project root)
# Available at: https://github.com/gabolsgabs/DALI
```

### Running the Code

**1. Prepare Training Data:**
```bash
python backend/scripts/prepare_segmented_manifest.py
```
This creates aligned 15-second audio segments from DALI annotations.

**2. Train Model:**
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python backend/training/train.py \
    --manifest backend/data/training_sets/dali_segmented_train.jsonl \
    --epochs 20 \
    --batch_size 2
```

**3. Run Benchmarks:**
```bash
python benchmark_models.py
```
Evaluates DTW, HMM, and CTC on available songs.

## Results Summary

- **Implementation**: All three methods fully implemented and functional
- **Training**: Successfully completed 1 epoch on 1,634 segments
- **Checkpoints**: Saved in `backend/outputs/checkpoints/`
- **Challenges**: Data loading bottleneck prevented full training completion on consumer hardware

See **`project proposal.txt`** for detailed results and analysis.

## Dataset

This project uses DALI v1.0:
- 5,358 songs with word-level annotations
- English-language pop music
- YouTube audio URLs (100 songs successfully downloaded)
- Ground truth word timestamps for evaluation

Dataset: Meseguer-Brocal et al., "DALI: A large dataset of synchronized audio, lyrics and notes," 2019.

## Citation

If referencing this work:

```
M. E. Aykuyt and M. Daşkaya, "A Comparative Study of Lyric-Audio Alignment Methods: 
From Dynamic Time Warping to Deep Learning Approaches," Course Project Report, 2026.
```

## Authors

- Murat Emirhan Aykuyt (1731100)
- Mehmet Daşkaya (2003445)

## License

This is an academic project. Code implementations use standard open-source libraries (PyTorch, Transformers, librosa). DALI dataset has its own licensing terms.
