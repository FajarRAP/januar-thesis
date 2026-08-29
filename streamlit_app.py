import streamlit as st
import pandas as pd
import numpy as np
import traceback
from joblib import load
from pathlib import Path
from utils.knn import KNN
from utils.dataset import Dataset
from utils.helpers import min_max_normalization

st.set_page_config(page_title="Deteksi Diabetes - KNN", page_icon="🩺", layout="wide")

ARTIFACTS = Path(__file__).parent.resolve()
model_path = ARTIFACTS / "best_knn_model.joblib"
dataframe = pd.read_csv(ARTIFACTS / "dataset_preprocessed.csv")
dataset = Dataset(dataframe)

@st.cache_resource
def load_artifacts() -> tuple[KNN, np.ndarray, np.ndarray]:
    try:
        model = load(model_path)
        predictions = model.predict(model.X_test.values)
        y_pred = predictions['Euclidean']

        return model, y_pred, model.y_test.values
    except Exception as e:
        print("Error loading artifacts:", e)
        traceback.print_exc()
        return KNN(), np.array([]), np.array([])

model, y_pred, y_test = load_artifacts()

st.title("🩺 Deteksi Diabetes (KNN)")
st.caption("Model ini adalah alat bantu dan **bukan** pengganti diagnosis tenaga medis.")

tab1 = st.tabs(["🔍 Prediksi Individu"])

st.subheader("Masukkan Data Pasien")

with st.container():
    st.markdown("### 👤 Informasi Umum")
    col1, col2, col3 = st.columns(3)
    with col1:
        umur = st.number_input("Umur (Tahun)", value=0, step=1, min_value=0, max_value=120)
    with col2:
        jenis_kelamin = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"), index=None, placeholder="Pilih jenis kelamin")
    with col3:
        tinggi = st.number_input("Tinggi Badan (m)", value=0.0, step=0.01, min_value=0.0, max_value=3.0, help="Contoh: 1.65")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        berat = st.number_input("Berat Badan (kg)", value=0.0, step=0.1, min_value=0.0, max_value=300.0)
    with col2:
        # Hitung BMI secara otomatis jika tinggi dan berat sudah diisi
        bmi_value = 0.0
        if tinggi > 0 and berat > 0:
            bmi_value = berat / (tinggi ** 2)
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", value=float(f"{bmi_value:.2f}"), disabled=True)
    with col3:
        pass

with st.container():
    st.markdown("### 🩸 Data Medis")
    col1, col2, col3 = st.columns(3)
    with col1:
        sistolik = st.number_input("Tekanan Darah Sistolik (mmHg)", value=0, step=1)
        diastolik = st.number_input("Tekanan Darah Diastolik (mmHg)", value=0, step=1)
    with col2:
        glukosa = st.number_input("Kadar Glukosa (mmol/L)", value=0.0, step=0.1)
        pulse_rate = st.number_input("Detak Jantung (Pulse Rate bpm)", value=0, step=1)
    with col3:
        pass

with st.container():
    st.markdown("### ⚕️ Riwayat Penyakit")
    col1, col2, col3 = st.columns(3)
    with col1:
        family_diabetes = st.checkbox("Ada Riwayat Diabetes di Keluarga?")
        hypertensive = st.checkbox("Menderita Hipertensi?")
    with col2:
        family_hypertension = st.checkbox("Ada Riwayat Hipertensi di Keluarga?")
        cardiovascular_disease = st.checkbox("Menderita Penyakit Kardiovaskular?")
    with col3:
        stroke = st.checkbox("Pernah Mengalami Stroke?")
        
if st.button("Prediksi", type="primary", use_container_width=True):
    if jenis_kelamin is None:
        st.warning("⚠️ Mohon pilih jenis kelamin terlebih dahulu!")
    else:
        # Setup Min-Max values
        X_min_max = {
            'umur': (dataset.X_min['umur'], dataset.X_max['umur']),
            'pulse_rate': (dataset.X_min['pulse_rate'], dataset.X_max['pulse_rate']),
            'sistolik': (dataset.X_min['sistolik'], dataset.X_max['sistolik']),
            'diastolik': (dataset.X_min['diastolik'], dataset.X_max['diastolik']),
            'glukosa': (dataset.X_min['glukosa'], dataset.X_max['glukosa']),
            'height': (dataset.X_min['height'], dataset.X_max['height']),
            'weight': (dataset.X_min['weight'], dataset.X_max['weight']),
            'bmi': (dataset.X_min['bmi'], dataset.X_max['bmi']),
            'family_diabetes': (dataset.X_min['family_diabetes'], dataset.X_max['family_diabetes']),
            'hypertensive': (dataset.X_min['hypertensive'], dataset.X_max['hypertensive']),
            'family_hypertension': (dataset.X_min['family_hypertension'], dataset.X_max['family_hypertension']),
            'cardiovascular_disease': (dataset.X_min['cardiovascular_disease'], dataset.X_max['cardiovascular_disease']),
            'stroke': (dataset.X_min['stroke'], dataset.X_max['stroke'])
        }
        
        # Transform and Normalize inputs
        transformed_gender = 1 if jenis_kelamin == "Perempuan" else 0
        norm_umur = min_max_normalization(umur, *X_min_max['umur'])
        norm_pulse_rate = min_max_normalization(pulse_rate, *X_min_max['pulse_rate'])
        norm_sistolik = min_max_normalization(sistolik, *X_min_max['sistolik'])
        norm_diastolik = min_max_normalization(diastolik, *X_min_max['diastolik'])
        norm_glukosa = min_max_normalization(glukosa, *X_min_max['glukosa'])
        norm_height = min_max_normalization(tinggi, *X_min_max['height'])
        norm_weight = min_max_normalization(berat, *X_min_max['weight'])
        norm_bmi = min_max_normalization(bmi_value, *X_min_max['bmi'])
        
        norm_family_diabetes = min_max_normalization(1 if family_diabetes else 0, *X_min_max['family_diabetes'])
        norm_hypertensive = min_max_normalization(1 if hypertensive else 0, *X_min_max['hypertensive'])
        norm_family_hypertension = min_max_normalization(1 if family_hypertension else 0, *X_min_max['family_hypertension'])
        norm_cardiovascular_disease = min_max_normalization(1 if cardiovascular_disease else 0, *X_min_max['cardiovascular_disease'])
        norm_stroke = min_max_normalization(1 if stroke else 0, *X_min_max['stroke'])
        
        # Urutan fitur harus sesuai dengan yang di-train di cell 81
        # 'transformed_gender', 'norm_umur', 'norm_pulse_rate', 'norm_sistolik', 'norm_diastolik', 
        # 'norm_glukosa', 'norm_height', 'norm_weight', 'norm_bmi', 'norm_family_diabetes', 
        # 'norm_hypertensive', 'norm_family_hypertension', 'norm_cardiovascular_disease', 'norm_stroke'
        
        new_data = np.array([[
            transformed_gender, norm_umur, norm_pulse_rate, norm_sistolik, norm_diastolik,
            norm_glukosa, norm_height, norm_weight, norm_bmi, norm_family_diabetes,
            norm_hypertensive, norm_family_hypertension, norm_cardiovascular_disease, norm_stroke
        ]])
        
        prediction = model.predict(new_data)['Euclidean']
        
        st.divider()
        if prediction == 1:
            st.error("### 🚨 Hasil Prediksi: Positif Diabetes Melitus (Tipe 2)")
            st.write("Disarankan untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan penanganan yang tepat.")
        else:
            st.success("### ✅ Hasil Prediksi: Negatif (Tidak Diabetes)")
            st.write("Terus pertahankan pola hidup sehat. Jangan lupa lakukan pemeriksaan rutin.")
            
        st.info("**Catatan:** Prediksi ini hanya berdasarkan model komputasi KNN dan tidak mempertimbangkan semua faktor medis. Konsultasikan dengan tenaga medis untuk diagnosis resmi.")

st.divider()
st.markdown("**Disclaimer:** Aplikasi ini bersifat edukatif dan sebagai _decision support_. Untuk diagnosis akhir, konsultasikan dengan tenaga kesehatan profesional.")
