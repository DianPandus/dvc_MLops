
# Laporan Eksperimen MLOps — MLflow & DVC (Penguins)

**Nama:** Dian Pandu Syahfitra  
**Mata Kuliah / Modul:** MLOps  
**Tanggal eksperimen:** (isi)  
**Repositori:** (isi link GitHub Anda)

## 1. Tujuan
- Membangun pipeline ML end‑to‑end dengan DVC.
- Melacak eksperimen & model menggunakan MLflow.
- Menjalankan CI otomatis via GitHub Actions.

## 2. Dataset
- **Sumber:** Palmer Penguins (CSV di GitHub).  
- **Target:** `species` (Adelie / Chinstrap / Gentoo).  
- **Fitur:** numerik (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) dan kategorikal (sex, island).

## 3. Arsitektur & Alur
- DVC stages: get_data → split_data → train → eval.
- MLflow: logging params, metrics, artifacts; local registry `mlruns/`.
- GitHub Actions: jalankan `dvc repro`, upload `metrics.json`, `model.joblib`, dan `mlruns/` sebagai artifacts.

## 4. Eksperimen
- **Model:** RandomForestClassifier (baseline).  
- **Hyperparameter kunci:** `n_estimators`, `max_depth`, `min_samples_split`.  
- **Metric evaluasi:** accuracy, macro F1.

Rangkuman hasil (isi setelah dijalankan):
- Accuracy: …
- F1 macro: …
- Confusion matrix (opsional): …

## 5. Diskusi
- Analisis error & fitur penting (feature importance).
- Bandingkan beberapa run (ubah hyperparameter di `params.yaml`).
- Catat perubahan dan efeknya di MLflow.

## 6. CI/CD
- Tunjukkan screenshot GitHub Actions berhasil.
- Jelaskan manfaat otomatisasi: reproduksibilitas, deteksi regresi, artifacts terarsip.

## 7. Kesimpulan & Pekerjaan Lanjutan
- Ringkasan pencapaian & rencana perbaikan (hyperparameter tuning, data checks, model registry remote, dsb.).
