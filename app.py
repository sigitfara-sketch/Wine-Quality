import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Kualitas Anggur Merah",
    page_icon="🍷",
    layout="wide"
)

# --- Judul dan Deskripsi ---
st.title("🍷 Prediksi Kualitas Anggur Merah")
st.markdown("""
Aplikasi ini menggunakan model **Random Forest Classifier** yang telah dilatih
untuk memprediksi apakah suatu anggur merah memiliki kualitas **Tinggi (>= 7)** atau **Rendah (< 7)**,
berdasarkan 11 fitur fisika-kimia.
""")
st.markdown("---")

# --- Muat Model dan Scaler ---
try:
    # Model dan Scaler harus berada di folder yang sama dengan app.py atau dapat diakses
    model = joblib.load('wine_quality_model.joblib')
    scaler = joblib.load('wine_quality_scaler.joblib')
    feature_names = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
except FileNotFoundError:
    st.error("Error: Pastikan file `wine_quality_model.joblib` dan `wine_quality_scaler.joblib` ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model atau scaler: {e}")
    st.stop()

# --- Input Pengguna dengan Kolom yang Menarik ---

st.header("Masukkan Parameter Anggur")

# Definisi rentang dan nilai default untuk setiap fitur (berdasarkan statistik data asli)
# Ini membantu pengguna memasukkan nilai yang realistis.
feature_ranges = {
    'fixed acidity': (4.0, 16.0, 8.3),
    'volatile acidity': (0.1, 1.6, 0.5),
    'citric acid': (0.0, 1.0, 0.3),
    'residual sugar': (0.9, 15.5, 2.5),
    'chlorides': (0.01, 0.65, 0.08),
    'free sulfur dioxide': (1.0, 72.0, 15.8),
    'total sulfur dioxide': (6.0, 289.0, 46.4),
    'density': (0.99, 1.004, 0.996),
    'pH': (2.7, 4.0, 3.3),
    'sulphates': (0.3, 2.0, 0.66),
    'alcohol': (8.4, 14.9, 10.4)
}

# Membuat 4 kolom untuk penempatan input yang lebih ringkas
col1, col2, col3, col4 = st.columns(4)

input_data = {}

with col1:
    input_data['fixed acidity'] = st.slider("Fixed Acidity", min_value=feature_ranges['fixed acidity'][0], max_value=feature_ranges['fixed acidity'][1], value=feature_ranges['fixed acidity'][2], step=0.1)
    input_data['volatile acidity'] = st.slider("Volatile Acidity", min_value=feature_ranges['volatile acidity'][0], max_value=feature_ranges['volatile acidity'][1], value=feature_ranges['volatile acidity'][2], step=0.01)
    input_data['citric acid'] = st.slider("Citric Acid", min_value=feature_ranges['citric acid'][0], max_value=feature_ranges['citric acid'][1], value=feature_ranges['citric acid'][2], step=0.01)

with col2:
    input_data['residual sugar'] = st.slider("Residual Sugar", min_value=feature_ranges['residual sugar'][0], max_value=feature_ranges['residual sugar'][1], value=feature_ranges['residual sugar'][2], step=0.1)
    input_data['chlorides'] = st.slider("Chlorides", min_value=feature_ranges['chlorides'][0], max_value=feature_ranges['chlorides'][1], value=feature_ranges['chlorides'][2], step=0.001, format="%.3f")
    input_data['free sulfur dioxide'] = st.slider("Free Sulfur Dioxide", min_value=feature_ranges['free sulfur dioxide'][0], max_value=feature_ranges['free sulfur dioxide'][1], value=feature_ranges['free sulfur dioxide'][2], step=0.1)

with col3:
    input_data['total sulfur dioxide'] = st.slider("Total Sulfur Dioxide", min_value=feature_ranges['total sulfur dioxide'][0], max_value=feature_ranges['total sulfur dioxide'][1], value=feature_ranges['total sulfur dioxide'][2], step=1.0)
    input_data['density'] = st.slider("Density", min_value=feature_ranges['density'][0], max_value=feature_ranges['density'][1], value=feature_ranges['density'][2], step=0.0001, format="%.4f")
    input_data['pH'] = st.slider("pH", min_value=feature_ranges['pH'][0], max_value=feature_ranges['pH'][1], value=feature_ranges['pH'][2], step=0.01)

with col4:
    input_data['sulphates'] = st.slider("Sulphates", min_value=feature_ranges['sulphates'][0], max_value=feature_ranges['sulphates'][1], value=feature_ranges['sulphates'][2], step=0.01)
    input_data['alcohol'] = st.slider("Alcohol", min_value=feature_ranges['alcohol'][0], max_value=feature_ranges['alcohol'][1], value=feature_ranges['alcohol'][2], step=0.1)


# --- Fungsi Prediksi ---
def predict_wine_quality(data, model, scaler, feature_names):
    # Buat DataFrame dari input
    input_df = pd.DataFrame([data])
    
    # Pastikan urutan kolom sesuai dengan yang digunakan saat pelatihan
    input_df = input_df[feature_names]

    # Skalakan data input
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    return prediction, prediction_proba

# --- Tombol dan Hasil Prediksi ---
st.markdown("---")
if st.button("🔬 Prediksi Kualitas Anggur"):
    
    # Lakukan prediksi
    prediction, prediction_proba = predict_wine_quality(input_data, model, scaler, feature_names)

    st.subheader("🎉 Hasil Prediksi")
    
    # Tampilkan hasil
    if prediction == 1:
        st.success(f"**Kualitas Anggur: TINGGI (High Quality)**")
        st.balloons()
    else:
        st.error(f"**Kualitas Anggur: RENDAH (Low Quality)**")
    
    # Tampilkan probabilitas
    st.markdown(f"""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
        **Probabilitas Kualitas Rendah (0):** {prediction_proba[0]*100:.2f}% <br>
        **Probabilitas Kualitas Tinggi (1):** {prediction_proba[1]*100:.2f}%
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")


# --- Analisis Feature Importance ---
st.header("📊 Analisis Fitur (Feature Importance)")

try:
    import plotly.express as px
    
    # Dapatkan feature importance dari model Random Forest
    importances = model.feature_importances_
    
    # Buat DataFrame untuk visualisasi
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Visualisasi dengan Plotly
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Tingkat Kepentingan (Feature Importance) dalam Prediksi',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Tealgrn,
        template='plotly_white'
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    *Grafik ini menunjukkan seberapa besar kontribusi setiap fitur dalam penentuan kualitas anggur oleh model.*
    *Fitur dengan Importance tertinggi adalah yang paling memengaruhi hasil prediksi.*
    """)

except ImportError:
    st.warning("Install `plotly` untuk menampilkan grafik Feature Importance (`pip install plotly`).")

st.markdown("---")
st.caption("Dibuat dengan Python, Scikit-learn, dan Streamlit.")