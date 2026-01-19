# Lyric-Audio Alignment System

Comparison of DTW, HMM, and Deep Learning (CTC) based alignment methods.

## Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (for audio processing)

## Installation

### Backend (Python)
```bash
# Environment setup (in project root)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend (Next.js)
```bash
cd frontend
npm install
```

## DALI Veri Seti Kurulumu (Opsiyonel)
Otomatik veri seti indirme ve test için:
```bash
cd backend
# DALI veri setini indirip dev-set oluşturur (Audio + Lyrics + Alignment)
python scripts/prepare_dali.py
```
Bu işlem `backend/data/dali` klasörüne örnek şarkıları indirecektir.

## Running the Application

1. **Start Backend Server:**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn main:app --reload
   ```
   Server will run at `http://localhost:8000`

2. **Start Frontend Client:**
   ```bash
   # In a new terminal
   cd frontend
   npm run dev
   ```
   Client will run at `http://localhost:3000`

## Nasıl Test Edilir?

### Yöntem 1: Kendi Dosyanızla
1. Uygulamayı açın: `http://localhost:3000`
2. **Upload Source** kısmından bir `.mp3` dosyası seçin.
3. Şarkı sözlerini "Lyrics" kutusuna yapıştırın.
4. **Upload & Initialize** butonuna basın. (Dosya yüklenip vocal separation başlayacak)
5. İşlem bitince, sağ tarafta **Run Alignment** altındaki modellerden birine (Örn: `DTW Baseline`) tıklayın.
6. Sonuçlar gelince **Audio Player**'dan oynatın ve **Karaoke View**'da takibi izleyin.

### Yöntem 2: DALI Veri Seti ile (Otomatik)
1. `python scripts/prepare_dali.py` çalıştırdıysanız `backend/data/dali` klasöründe hazır dosyalar olacaktır.
2. Bu dosyalardan birini (mp3) ve ilgili json içindeki lyrics text'ini kullanarak Yöntem 1'i uygulayabilirsiniz.

## Features
- **3 Alignment Models:** DTW (Baseline), HMM (Phoneme), CTC (Deep Learning)
- **Source Separation:** Automatic vocal separation using Demucs
- **Karaoke View:** Real-time word highlighting & Click-to-seek
- **Comparison View:** Visual analysis of alignment accuracy
