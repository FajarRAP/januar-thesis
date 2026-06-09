

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Deteksi Diabetes - KNN", page_icon="🩺", layout="wide")

ARTIFACTS = Path(__file__).parent.resolve()

@st.cache_resource

def _fallback_train_and_save(artifacts_dir: Path):
    import pandas as pd
    import numpy as np
    import json, joblib
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    data_path_csv = artifacts_dir / "diabetes.csv"
    if not data_path_csv.exists():
        raise FileNotFoundError("File 'diabetes.csv' tidak ditemukan untuk pelatihan ulang otomatis.")

    df = pd.read_csv(data_path_csv, sep=";")
    df.columns = [c.strip() for c in df.columns]
    # buang kolom unnamed kosong
    for c in list(df.columns):
        if c.lower().startswith("unnamed"):
            if df[c].isna().all() or (df[c].astype(str).str.strip()=="").all():
                df = df.drop(columns=[c])

    target_col = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski"))
    ])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_balance": y.value_counts(normalize=True).to_dict()
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    metrics["cv_auc_mean"] = float(cv_auc.mean())
    metrics["cv_auc_std"] = float(cv_auc.std())

    joblib.dump(pipe, artifacts_dir / "model_knn.pkl")
    meta = {"features": X.columns.tolist(), "target": target_col, "metrics": metrics}
    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta))
    return pipe, meta


def load_artifacts():
    try:
        model = joblib.load(ARTIFACTS / "model_knn.pkl")
        meta = json.loads((ARTIFACTS / "model_meta.json").read_text())
        return model, meta
    except Exception as e:
        st.warning(f"Gagal memuat model tersimpan (kemungkinan beda versi scikit-learn): {str(e)}. Aplikasi akan melatih ulang otomatis.")
        model, meta = _fallback_train_and_save(ARTIFACTS)
        return model, meta

model, meta = load_artifacts()
FEATURES = meta["features"]
TARGET = meta["target"]
metrics = meta["metrics"]

st.title("🩺 Deteksi Diabetes (KNN)")
st.caption("Model ini adalah alat bantu dan **bukan** pengganti diagnosis tenaga medis.")

tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Individu", "📄 Prediksi Batch (CSV)", "📈 Evaluasi Model"])

with tab1:
    st.subheader("Masukkan Nilai Fitur")
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            # tentukan batasan default yang wajar
            dtype = float
            step = 1.0
            default = 0.0
            inputs[feat] = st.number_input(feat, value=default, step=step, format="%.4f")
    if st.button("Prediksi"):
        X = pd.DataFrame([inputs])[FEATURES]
        proba = model.predict_proba(X)[0,1]
        pred = int(proba >= 0.5)
        st.metric("Probabilitas Diabetes", f"{proba:.3f}")
        st.metric("Kelas Prediksi", "1 (Diabetes)" if pred==1 else "0 (Tidak Diabetes)")
        st.info("Keputusan dibuat dengan K-Nearest Neighbors menggunakan pipeline: imputasi median + standardisasi.")

with tab2:
    st.subheader("Unggah CSV untuk Prediksi Massal")
    st.write("CSV harus memuat kolom: **" + ", ".join(FEATURES) + "**")
    file = st.file_uploader("Pilih file CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error("Kolom berikut tidak ditemukan di CSV: " + ", ".join(missing))
        else:
            X = df[FEATURES]
            proba = model.predict_proba(X)[:,1]
            pred = (proba >= 0.5).astype(int)
            out = df.copy()
            out["prob_diabetes"] = proba
            out["pred_diabetes"] = pred
            st.success(f"Berhasil memproses {len(df)} baris.")
            st.dataframe(out.head(20))
            st.download_button("Unduh Hasil (CSV)", out.to_csv(index=False).encode("utf-8"), "prediksi_diabetes.csv", "text/csv")

with tab3:
    st.subheader("Ringkasan Performa (Hold-out Test + CV)")
    m = metrics
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Accuracy", f"{m['accuracy']:.3f}")
        st.metric("Precision", f"{m['precision']:.3f}")
    with colB:
        st.metric("Recall (Sensitivitas)", f"{m['recall']:.3f}")
        st.metric("F1", f"{m['f1']:.3f}")
    with colC:
        st.metric("AUC-ROC (Test)", f"{m['auc']:.3f}")
        st.metric("AUC-ROC (CV mean ± std)", f"{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}")

    import numpy as np
    cm = np.array(m["confusion_matrix"])
    st.write("**Confusion Matrix (threshold 0.5)**")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

    st.caption(f"N train: {m['n_train']} • N test: {m['n_test']} • Kelas (0/1): {m['class_balance']}")
    
st.divider()
st.markdown("**Disclaimer:** Aplikasi ini bersifat edukatif dan sebagai _decision support_. Untuk diagnosis akhir, konsultasikan dengan tenaga kesehatan profesional.")
