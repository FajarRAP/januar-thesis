# Laporan Evaluasi Singkat (KNN Diabetes)

**Target**: Outcome  
**Fitur**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, Diabetes PedigreeFunction, Age, Unnamed: 10

## Metrik Hold-out (Test)
- Accuracy: 0.993
- Precision: 0.979
- Recall: 1.000
- F1: 0.989
- AUC-ROC: 0.994

**Confusion Matrix (threshold 0.5)**  
[[260   3]
 [  0 137]]

## Cross-Validation (5-fold)
- AUC (mean ± std): 0.986 ± 0.009

## Catatan
- Pipeline: Imputer(median) → StandardScaler → KNN(k=7, weights=distance).
- Data dibaca dengan delimiter `;`. Kolom 'Unnamed' kosong dihapus.
- Kelas: {0: 0.658, 1: 0.342}
