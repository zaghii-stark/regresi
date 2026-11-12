import streamlit as st
import pandas as pd
import numpy as np

def baca_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_excel(file)

def bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep):
    data = df[kolom_fitur + [kolom_target]].apply(pd.to_numeric, errors="raise")
    if data.isnull().any().any():
        raise ValueError("Terdapat sel kosong/NaN pada kolom yang dipilih.")
    
    X = data[kolom_fitur].to_numpy(dtype=float)
    y = data[kolom_target].to_numpy(dtype=float)
    
    nama_w = [f"w{i+1}" for i in range(len(kolom_fitur))]
    if pakai_intersep:
        X = np.column_stack([np.ones(len(X)), X])
        nama_w = ["w0"] + nama_w
        
    return X, y, nama_w

def selesaikan_solver_python(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError(
            "Matriks XᵀX singular (tidak punya invers). Kemungkinan ada fitur yang "
            "saling bergantung linear atau jumlah sampel kurang dari jumlah parameter."
        )
    
    w = np.linalg.solve(XtX, Xty)
    return XtX, Xty, w

st.set_page_config(page_title="SPL Regresi Linear (Solver Python)", layout="wide")
st.title("Penyelesaian SPL dengan Solver Bawaan Python")
st.caption("Metode: Solver Bawaan Python (np.linalg.solve) - Khusus Digit NIM 8 dan 9")

file = st.file_uploader("Unggah file data (CSV / Excel)", type=["csv", "xlsx", "xls"])

if file is not None:
    try:
        df = baca_file(file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.subheader("1. Data yang Diunggah")
    st.dataframe(df)

    kolom_numerik = df.select_dtypes(include="number").columns.tolist()
    if len(kolom_numerik) < 2:
        st.error("Data harus memiliki minimal 2 kolom numerik (fitur dan target).")
        st.stop()

    st.subheader("2. Pilih Variabel")
    col1, col2 = st.columns(2)
    with col1:
        kolom_target = st.selectbox(
            "Kolom target (y)", kolom_numerik, index=len(kolom_numerik) - 1
        )
    with col2:
        abaikan = ("sampel", "no", "id", "nomor")
        default_fitur = [
            k for k in kolom_numerik
            if k != kolom_target and k.lower() not in abaikan
        ]
        kolom_fitur = st.multiselect(
            "Kolom fitur (x)", [k for k in kolom_numerik if k != kolom_target],
            default=default_fitur,
        )

    pakai_intersep = st.checkbox("Tambahkan intersep (w0)", value=True)

    if st.button("Selesaikan (Solver Python)", type="primary"):
        try:
            if not kolom_fitur:
                raise ValueError("Pilih minimal satu kolom fitur.")
            
            X, y, nama_w = bangun_matriks(df, kolom_fitur, kolom_target, pakai_intersep)
            XtX, Xty, w = selesaikan_solver_python(X, y)
            
            st.session_state["hasil"] = dict(
                file=file.name, fitur=kolom_fitur, target=kolom_target,
                X=X, y=y, nama_w=nama_w, XtX=XtX, Xty=Xty, w=w,
                intersep=pakai_intersep,
            )
        except Exception as e:
            st.session_state.pop("hasil", None)
            st.error(f"Terjadi kesalahan: {e}")

    h = st.session_state.get("hasil")
    if h and h["file"] == file.name:
        X, y, w, nama_w = h["X"], h["y"], h["w"], h["nama_w"]
        w = w + 0.0

        st.subheader("3. Langkah Perhitungan Matriks")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matriks Desain X**")
            st.dataframe(pd.DataFrame(X, columns=nama_w if h["intersep"] else h["fitur"]))
        with c2:
            st.markdown("**Vektor y**")
            st.dataframe(pd.DataFrame(y, columns=[h["target"]]))

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Matriks (XᵀX)**")
            st.dataframe(pd.DataFrame(h["XtX"], index=nama_w, columns=nama_w))
        with c4:
            st.markdown("**Vektor (Xᵀy)**")
            st.dataframe(pd.DataFrame(h["Xty"], index=nama_w, columns=["Xᵀy"]))

        st.subheader("4. Solusi Bobot w (Hasil Solver Python)")
        hasil_w = pd.DataFrame({"Parameter": nama_w, "Nilai Bobot": np.round(w, 4)})
        st.dataframe(hasil_w, hide_index=True)

        suku = []
        for nama, nilai, fitur in zip(nama_w, w, ([None] + h["fitur"]) if h["intersep"] else h["fitur"]):
            suku.append(f"{nilai:.4f}" if fitur is None else f"{nilai:.4f}·{fitur}")
        st.info(f"Model SPL: {h['target']} = " + " + ".join(suku))

        y_hat = X @ w
        galat = y - y_hat
        sse = float(np.sum(galat ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst if sst > 0 else float("nan")

        m1, m2 = st.columns(2)
        m1.metric("RMSE", f"{np.sqrt(np.mean(galat ** 2)):.4f}")
        m2.metric("R²", f"{r2:.4f}")

        st.markdown("**Tabel Perbandingan Aktual vs Prediksi**")
        df_pred = pd.DataFrame({"Aktual": y, "Prediksi": np.round(y_hat, 4), "Galat": np.round(galat, 4)})
        st.dataframe(df_pred)

        st.subheader("5. Prediksi Data Baru")
        nilai_baru = []
        kolom_in = st.columns(len(h["fitur"]))
        for kol, nama in zip(kolom_in, h["fitur"]):
            nilai_baru.append(kol.number_input(nama, value=float(df[nama].mean())))
        
        x_baru = np.array(([1.0] if h["intersep"] else []) + nilai_baru)
        st.success(f"Prediksi Nilai {h['target']} = {float(x_baru @ w):.2f}")
else:    st.info("Silakan unggah file CSV/Excel terlebih dahulu.")