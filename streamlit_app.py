import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from utils.knn import KNN
from utils.metrics_evaluator import MetricsEvaluator
from utils.helpers import to_percent

st.set_page_config(page_title="Deteksi Diabetes - KNN", page_icon="🩺", layout="wide")

ARTIFACTS = Path(__file__).parent.resolve()

@st.cache_resource

def split_dataset(dataframe: pd.DataFrame, test_size: float = .2):
    training_count = int(dataframe.shape[0] * (1 - test_size))
    
    # Memisahkan fitur (X) dan target (y)
    X = dataframe[['Norm_sistolik', 'Norm_diastolik', 'Norm_umur', 'Norm_gds', 'transformed_gender']]
    y = dataframe['Diagnosa']
    
    # Membagi dataset menjadi data Latih (Train) dan data Uji (Test)
    X_train = X.iloc[:training_count]
    y_train = y.iloc[:training_count]
    X_test = X.iloc[training_count:]
    y_test = y.iloc[training_count:]
    
    return X_train, y_train, X_test, y_test

def load_artifacts() -> MetricsEvaluator:
    X_train, y_train, X_test, y_test = split_dataset(pd.read_csv(ARTIFACTS / "dataset_preprocessed.csv"))
    
    try:
        model = KNN()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test.values)
        y_pred = predictions['Euclidean']
        evaluator = MetricsEvaluator(y_pred, y_test.values)

        return evaluator
    except Exception as e:
        print("Error loading artifacts:", e)
        return MetricsEvaluator(np.array([]), np.array([]))

model = load_artifacts()

st.title("🩺 Deteksi Diabetes (KNN)")
st.caption("Model ini adalah alat bantu dan **bukan** pengganti diagnosis tenaga medis.")

tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Individu", "📄 Prediksi Batch (CSV)", "📈 Evaluasi Model"])

with tab1:
    st.subheader("Masukkan Nilai Fitur")
    cols = st.columns(3)
    inputs = {}
    with cols[0]:
        inputs['year'] = st.number_input("Umur", value=0, step=1)
        inputs['systolic'] = st.number_input("Tekanan Darah Sistolik", value=0, step=1)
    with cols[1]:
        inputs['gender'] = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"), index=None, placeholder="Pilih jenis kelamin")
        inputs['diastolic'] = st.number_input("Tekanan Darah Diastolik", value=0, step=1)
    with cols[2]:
        inputs['random_blood_sugar'] = st.number_input("Gula Darah Sewaktu", value=0, step=1)
            
    if st.button("Prediksi"):
        # TODO: Rumus KNN dari Google Colab
        st.info("Keputusan dibuat dengan K-Nearest Neighbors menggunakan pipeline: imputasi median + standardisasi.")

with tab2:
    st.subheader("Unggah CSV untuk Prediksi Massal")
#     st.write("CSV harus memuat kolom: **" + ", ".join(FEATURES) + "**")
#     file = st.file_uploader("Pilih file CSV", type=["csv"])
#     if file is not None:
#         df = pd.read_csv(file)
#         missing = [c for c in FEATURES if c not in df.columns]
#         if missing:
#             st.error("Kolom berikut tidak ditemukan di CSV: " + ", ".join(missing))
#         else:
#             X = df[FEATURES]
#             proba = model.predict_proba(X)[:,1]
#             pred = (proba >= 0.5).astype(int)
#             out = df.copy()
#             out["prob_diabetes"] = proba
#             out["pred_diabetes"] = pred
#             st.success(f"Berhasil memproses {len(df)} baris.")
#             st.dataframe(out.head(20))
#             st.download_button("Unduh Hasil (CSV)", out.to_csv(index=False).encode("utf-8"), "prediksi_diabetes.csv", "text/csv")

with tab3:
    st.subheader("Ringkasan Performa*")
    st.write("*Berdasarkan dataset skripsi")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Accuracy", to_percent(model.accuracy()))
        st.metric("Precision", to_percent(model.precision()))
    with colB:
        st.metric("Recall (Sensitivitas)", to_percent(model.recall()))
        st.metric("F1", to_percent(model.f1_score()))
    st.write("**Confusion Matrix**")
    st.pyplot(model.plot_confusion_matrix())
    
st.divider()
st.markdown("**Disclaimer:** Aplikasi ini bersifat edukatif dan sebagai _decision support_. Untuk diagnosis akhir, konsultasikan dengan tenaga kesehatan profesional.")
