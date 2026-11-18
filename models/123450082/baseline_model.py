import os
import json
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr

# Mengimpor modul pemotong sesuai protokol tim (untuk membuang 1 detik awal)
from src.protokol import pemotong

# 1. FUNGSI METRIK EVALUASI
def calculate_si_sdr(target, estimate, eps=1e-8):
    """Menghitung metrik SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)."""
    target = target - np.mean(target)
    estimate = estimate - np.mean(estimate)
    
    # Hitung faktor skala (alpha)
    alpha = np.sum(estimate * target) / (np.sum(target ** 2) + eps)
    target_scaled = alpha * target
    
    # Pisahkan komponen noise/distorsi
    noise = estimate - target_scaled
    
    # Hitung rasio dalam dB
    val = 10 * np.log10((np.sum(target_scaled ** 2) + eps) / (np.sum(noise ** 2) + eps))
    return val

def main():
    # 2. KONFIGURASI JALUR BERKAS (PATH)
    # Asumsi skrip dijalankan dari root folder: capstone-deep-learning
    label_path = "data/label.csv"
    index_path = "data/audio_index.csv"
    audio_dir = "data/audio"
    
    print("Memuat berkas CSV...")
    df_label = pd.read_csv(label_path)
    df_index = pd.read_csv(index_path)

    # 3. FILTER DATA UJI (P06) DAN PISAHKAN KONDISI
    # Hanya mengevaluasi Partisipan P06
    df_p06 = df_label[df_label['partisipan'] == 'P06']
    
    # Hening (Target Bersih): Kipas 0 dan Pompa 0
    df_bersih = df_p06[(df_p06['kipas'] == 0) & (df_p06['pompa'] == 0)]
    # Bising (Input Model): Selain kondisi di atas
    df_bising = df_p06[~((df_p06['kipas'] == 0) & (df_p06['pompa'] == 0))]

    # Fungsi pembantu untuk memotong 4 detik dari file 60 detik
    def ekstrak_audio(t_mulai_kejadian):
        for _, row_idx in df_index.iterrows():
            t_mulai_audio = row_idx['t_mulai_host']
            durasi_audio = row_idx['jumlah_sampel'] / row_idx['sample_rate']
            
            # Jika waktu kejadian berada di dalam rentang file audio ini
            if t_mulai_audio <= t_mulai_kejadian < (t_mulai_audio + durasi_audio):
                nama_file = row_idx['berkas']
                offset = t_mulai_kejadian - t_mulai_audio
                path_wav = os.path.join(audio_dir, nama_file)
                
                # Ekstrak 4 detik dari titik kejadian menggunakan librosa
                y, sr = librosa.load(path_wav, sr=16000, offset=offset, duration=4.0)
                
                # Terapkan protokol dari penjaga data (buang 1 detik pertama, sisa 3 detik)
                y_potong = pemotong.potong(y, sr=16000, durasi_target=3.0)
                return y_potong
        return None

    # 4. KELOMPOKKAN TARGET BERSIH BERDASARKAN KALIMAT
    # Agar audio bising dievaluasi dengan audio bersih yang kalimatnya sama
    print("Mengekstrak target audio bersih...")
    kamus_bersih = {}
    for _, row in df_bersih.iterrows():
        kalimat = str(row['kalimat']).strip().lower()
        audio_bersih = ekstrak_audio(row['t_mulai_host'])
        if audio_bersih is not None:
            kamus_bersih[kalimat] = audio_bersih

    # 5. BERSIHKAN AUDIO BISING DAN HITUNG SI-SDR
    print("Membersihkan bising (Wiener Filter) dan menghitung metrik...")
    skor_sdr = []
    
    for _, row in df_bising.iterrows():
        kalimat = str(row['kalimat']).strip().lower()
        audio_bising = ekstrak_audio(row['t_mulai_host'])
        
        # Pasangkan dengan target bersih yang kalimatnya sama
        if audio_bising is not None and kalimat in kamus_bersih:
            audio_target = kamus_bersih[kalimat]
            
            # Pastikan panjang array sama persis 
            min_len = min(len(audio_bising), len(audio_target))
            audio_bising = audio_bising[:min_len]
            audio_target = audio_target[:min_len]

            # Denoise sinyal bising
            prediksi = nr.reduce_noise(y=audio_bising, sr=16000, stationary=True)
            
            # Hitung skor SI-SDR
            skor = calculate_si_sdr(audio_target, prediksi)
            skor_sdr.append(skor)
    
    # 6. SIMPAN HASIL KE METRICS.JSON
    rata_sdr = float(np.mean(skor_sdr)) if skor_sdr else 0.0
    print(f"Selesai! Rata-rata SI-SDR Baseline: {rata_sdr:.4f} dB")
    
    out_dir = "results/baseline"
    os.makedirs(out_dir, exist_ok=True)
    hasil = {
        "model": "Spectral Subtraction / Wiener Filter",
        "metode": "non-DL baseline",
        "status_data": "uji_P06_sesi-20261117-071715",
        "metrics": {
            "si_sdr_peningkatan": round(rata_sdr, 4),
            "akurasi_perintah": 0.0 # 0.0 karena ini hanya model denoising, bukan klasifikasi teks
        }
    }
    
    output_path = os.path.join(out_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(hasil, f, indent=4)
    print(f"Batas bawah telah dicatat di {output_path}")

if __name__ == "__main__":
    main()