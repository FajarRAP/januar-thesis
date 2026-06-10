import streamlit as st
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from utils.knn import KNN
from utils.dataset import Dataset
from utils.metrics_evaluator import MetricsEvaluator
from utils.helpers import to_percent, min_max_normalization

st.set_page_config(page_title="Deteksi Diabetes - KNN", page_icon="🩺", layout="wide")

ARTIFACTS = Path(__file__).parent.resolve()
dataframe = pd.read_csv(ARTIFACTS / "dataset_preprocessed.csv")
dataset = Dataset(dataframe)

@st.cache_resource

def load_artifacts() -> tuple[KNN, np.ndarray, np.ndarray]:
    X_train, y_train, X_test, y_test = dataset.split(X=dataset.X[['Norm_sistolik', 'Norm_diastolik', 'Norm_umur', 'Norm_gds', 'transformed_gender']])
    
    try:
        model = KNN()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test.values)
        y_pred = predictions['Euclidean']

        return model, y_pred, y_test.values
    except Exception as e:
        print("Error loading artifacts:", e)
        traceback.print_exc()
        return KNN(), np.array([]), np.array([])

model, y_pred, y_test = load_artifacts()

st.title("🩺 Deteksi Diabetes (KNN)")
st.caption("Model ini adalah alat bantu dan **bukan** pengganti diagnosis tenaga medis.")

# "📄 Prediksi Batch (CSV)",
tab1, tab3 = st.tabs(["🔍 Prediksi Individu", "📈 Evaluasi Model"])

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
        X_min_max_values = {
            'year': (dataset.X_min['Umur'], dataset.X_max['Umur']),
            'systolic': (dataset.X_min['Sistolik'], dataset.X_max['Sistolik']),
            'diastolic': (dataset.X_min['Diastolik'], dataset.X_max['Diastolik']),
            'random_blood_sugar': (dataset.X_min['GDS'], dataset.X_max['GDS'])
        }
        
        # Transform and Normalize inputs
        transformed_gender = 1 if inputs['gender'] == "Perempuan" else 0
        normalized_year = min_max_normalization(inputs['year'], *X_min_max_values['year'])
        normalized_systolic = min_max_normalization(inputs['systolic'], *X_min_max_values['systolic'])
        normalized_diastolic = min_max_normalization(inputs['diastolic'], *X_min_max_values['diastolic'])
        normalized_random_blood_sugar = min_max_normalization(inputs['random_blood_sugar'], *X_min_max_values['random_blood_sugar'])
        
        new_data = np.array([[normalized_systolic, normalized_diastolic, normalized_year, normalized_random_blood_sugar, transformed_gender]])
        prediction = model.predict(new_data)['Euclidean']
        
        st.metric("Kelas Prediksi", "1 (Diabetes Melitus 2)" if(prediction == 1) else "0 (Tidak Diabetes)")
        st.info("**Catatan:** Prediksi ini hanya berdasarkan model KNN dan tidak mempertimbangkan faktor lain yang mungkin relevan. Konsultasikan dengan tenaga medis untuk diagnosis yang akurat.")

# with tab2:
    # st.subheader("Unggah CSV untuk Prediksi Massal")
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
    evaluator = MetricsEvaluator(y_pred, y_test)
    st.subheader("Ringkasan Performa*")
    st.write("*Berdasarkan dataset skripsi")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Accuracy", to_percent(evaluator.accuracy()))
        st.metric("Precision", to_percent(evaluator.precision()))
    with colB:
        st.metric("Recall (Sensitivitas)", to_percent(evaluator.recall()))
        st.metric("F1", to_percent(evaluator.f1_score()))
    st.write("**Confusion Matrix**")
    st.pyplot(evaluator.plot_confusion_matrix())
    
st.divider()
st.markdown("**Disclaimer:** Aplikasi ini bersifat edukatif dan sebagai _decision support_. Untuk diagnosis akhir, konsultasikan dengan tenaga kesehatan profesional.")
