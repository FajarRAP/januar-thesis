# Deteksi Diabetes (KNN) — Streamlit (Lokal)

## Cara Menjalankan Lokal
1. Buat environment (opsional) dan install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi:
   ```bash
   streamlit run streamlit_app.py
   ```

## Dataset
Menggunakan file `diabetes.csv` (format dipisah `;`). Kolom target terdeteksi otomatis: **Outcome**.

## Pipeline
- Imputasi median → StandardScaler → KNeighborsClassifier (`n_neighbors=7`, `weights=distance`).
- Split: 80/20 dengan stratifikasi, plus 5-fold CV untuk AUC.
- Metrik yang ditampilkan: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix.

## Catatan
- Aplikasi **tidak menyimpan** input.
- Model **bukan** alat diagnosis medis.


## Troubleshooting
Jika muncul error `EuclideanDistance`/versi scikit-learn saat memuat model, aplikasi akan **melatih ulang otomatis** memakai `diabetes.csv` di folder yang sama.
Pastikan file `diabetes.csv` tetap ada.
