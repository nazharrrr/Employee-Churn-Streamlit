import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from churn_module import DataLoader, AnalyzerData, ChurnModel

# Mengeluarkan Data
loader = DataLoader('employee_churn.csv')
df = loader.load_data()
df = loader.clean_data()

# Analisisi Data
analyzer=AnalyzerData(df)

# Modeling
model = ChurnModel(df)
x_train, x_test, y_train, y_test = model.prepare_data()
model.train(x_train, y_train)
accuracy = model.evaluate(x_test, y_test)

# Streamlit app
st.set_page_config("Employee Churn Dashboard", layout="wide")
st.sidebar.title('⚙️ Navigation')
page = st.sidebar.radio('Go to:', ("Overview", "Visualizations", "Model & Prediction", "Insight & Recommend"))

# Masuk ke laman Overview
if page == 'Overview':
  st.title('📊 Employee Churn Analysis Dashboard')
  st.write('Analisis perilaku karyawan dan prediksi kemungkinan churn.')

  st.write('### Dataset Overview')
  st.dataframe(df.head())

  col1, col2 = st.columns(2)
  with col1:
    st.write('### Churn Rate (%)')
    churn_rate = analyzer.churn_rate()
    st.metric(label='Churn Rate', value=f'{churn_rate:.2f}%')
  with col2:
    st.write('### Model Accuracy')
    st.metric(label='Accuracy', value=f'{accuracy*100:.2f}%')

  col3, col4 = st.columns(2)
  with col3:
    st.write('### Churn Rate by Age')
    churn_age = analyzer.churn_rate_age()
    st.dataframe(churn_age)
  with col4:
    st.write('### Churn Rate by Gender')
    churn_gender = analyzer.churn_rate_gender()
    st.dataframe(churn_gender)

  col5, col6 = st.columns(2)
  with col5:
    st.write('### Churn Rate by Marital Status')
    churn_marital = analyzer.churn_rate_marital()
    st.dataframe(churn_marital)
  with col6:
    st.write('### Churn Rate by Children')
    churn_children = analyzer.churn_rate_children()
    st.dataframe(churn_children)

  st.write('### Churn Rate by Performance')
  churn_performance = analyzer.churn_rate_performance()
  st.dataframe(churn_performance)

# Masuk ke Laman Visualisasi
elif page == 'Visualizations':
  st.title('📊 Visualizations')

  col1, col2 = st.columns(2)

# --- Distribusi Churn ---
  with col1:
    st.write('### Distribusi Churn')
    fig, ax = plt.subplots(figsize=(5, 4))

# Plot countplot churn
    sns.countplot(data=df, x='churn', ax=ax, palette='Set2')

# Ubah label sumbu X jadi Stay / Churn
    ax.set_xticklabels(['Stay', 'Churn'])

# Tambah angka di atas bar
    for p in ax.patches:
      ax.text(
        p.get_x() + p.get_width() / 2,  # posisi tengah bar
        p.get_height() + 1,             # posisi vertikal sedikit di atas bar
        f'{int(p.get_height())}',       # tampilkan jumlah
        ha='center', fontsize=10
    )

# Label dan judul
    ax.set_xlabel('Employee Status')
    ax.set_ylabel('Jumlah Karyawan')
    ax.set_title('Distribusi Churn', fontsize=13, weight='bold')

    st.pyplot(fig)

# --- Churn Rate by Performance ---
  with col2:
    st.write('### Churn Rate by Performance')
    churn_performance = analyzer.churn_rate_performance()
    fig, ax = plt.subplots(figsize=(5, 4))
    churn_performance.plot(kind='bar', color=['#E74C3C', '#3498DB'], ax=ax, legend=False)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_xlabel('Performance Level')
    ax.set_title('Churn Rate by Performance', fontsize=13, weight='bold')

    # Tambah label angka di atas bar
    for i, v in enumerate(churn_performance.values):
      ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

    st.pyplot(fig)


  col3, col4 = st.columns(2)

# --- Churn Rate by Age ---
  with col3:
    st.write('### Churn Rate by Age')
    churn_age = analyzer.churn_rate_age()
    fig, ax = plt.subplots(figsize=(5, 4))
    churn_age.plot(kind='bar', color=['#9B59B6', '#1ABC9C'], ax=ax, legend=False)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_xlabel('Kelompok Usia')
    ax.set_title('Churn Rate by Age', fontsize=13, weight='bold')

    for i, v in enumerate(churn_age.values):
      ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

    st.pyplot(fig)

# --- Churn Rate by Gender ---
  with col4:
    st.write('### Churn Rate by Gender')
    churn_gender = analyzer.churn_rate_gender()
    fig, ax = plt.subplots(figsize=(5, 4))
    churn_gender.plot(kind='bar', color=['#36A2EB', '#FF6384'], ax=ax, legend=False)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_xlabel('Gender')
    ax.set_title('Churn Rate by Gender', fontsize=13, weight='bold')

    for i, v in enumerate(churn_gender.values):
      ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

    st.pyplot(fig)

  col5, col6 = st.columns(2)

# --- Plot Churn Rate by Marital Status ---

  with col5:
    st.write('### Churn Rate by Marital Status')
    churn_marital = analyzer.churn_rate_marital()
    fig, ax = plt.subplots(figsize=(5, 4))
    churn_marital.plot(kind='bar', color=['#4C72B0', '#55A868'], ax=ax, legend=False)

    # Ganti label X-axis dan judul
    ax.set_xlabel('Marital Status', fontsize=11)
    ax.set_ylabel('Churn Rate (%)', fontsize=11)
    ax.set_title('Churn Rate by Marital Status', fontsize=13, weight='bold')

    # Set label kategori
    ax.set_xticks(range(len(churn_marital.index)))
    ax.set_xticklabels(['Single', 'Married'], rotation=0)

    # Tambah nilai di atas bar
    for i, v in enumerate(churn_marital.values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

    st.pyplot(fig)

# --- Plot Churn Rate by Children ---
  with col6:
    st.write('### Churn Rate by Children')
    churn_children = analyzer.churn_rate_children()
    fig, ax = plt.subplots(figsize=(5, 4))
    churn_children.plot(kind='bar', color=['#1F618D', '#F5B041'], ax=ax, legend=False)

    ax.set_xlabel('Has Children', fontsize=11)
    ax.set_ylabel('Churn Rate (%)', fontsize=11)
    ax.set_title('Churn Rate by Children', fontsize=13, weight='bold')

    ax.set_xticks(range(len(churn_children.index)))
    ax.set_xticklabels(['No', 'Yes'], rotation=0)

    for i, v in enumerate(churn_children.values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)

    st.pyplot(fig)


# Masuk laman Model Prediction
elif page == "Model & Prediction":
    st.title('🤖 Model & Prediction')
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30)
        tenure_months = st.slider("Tenure (Month)", 1, 60, 12)
        monthly_income = st.slider("Salary", 20000, 100000, 200000)
    with col2:
        has_marital_input = st.selectbox("Has Marital?", ('Yes', 'No'))
        has_marital = 1 if has_marital_input == 'Yes' else 0
        has_children_input = st.selectbox("Has Children?", ('Yes', 'No'))
        has_children = 1 if has_children_input == 'Yes' else 0
        performance_score = st.slider("Performance Rating", 1, 3, 5)

    if st.button('Cek'):
        input_data = {
            'age' : age,
            'tenure_months' : tenure_months,
            'monthly_income' : monthly_income,
            'has_marital' : has_marital,
            'has_children' : has_children,
            'performance_score' : performance_score
        }

        pred = model.predict(input_data)
        if pred == 1:
            st.error('Employee Will Churn')
        else:
            st.success('Employee Will Not Churn')

# Masuk ke Laman Insight dan Rekomendasi
# Masuk ke Laman Insight dan Rekomendasi
elif page == "Insight & Recommend":
    # ===== PAGE HEADER =====
    st.title("💡 Employee Churn — Insight & HR Recommendations")
    st.markdown("""
    <div style='text-align:justify'>
    Halaman ini menyajikan ringkasan temuan utama dari analisis *employee churn* beserta rekomendasi strategis
    yang dapat digunakan oleh tim HR untuk meningkatkan retensi karyawan dan mengoptimalkan strategi manajemen SDM.
    </div>
    """, unsafe_allow_html=True)

    # ===== DATA PREPARATION =====
    churn_rate = analyzer.churn_rate()
    churn_age = analyzer.churn_rate_age()
    churn_gender = analyzer.churn_rate_gender()
    churn_perf = analyzer.churn_rate_performance()
    churn_marital = analyzer.churn_rate_marital()
    churn_children = analyzer.churn_rate_children()

    # ===== KEY INSIGHTS SECTION =====
    st.markdown("## 📊 Key Insights")

    st.markdown(f"""
    <div style='background-color:#f8f9fa;padding:15px;border-radius:10px;border-left:5px solid #4B9CD3;margin-bottom:15px; color:#000000;'>
    <b>Total Churn Rate:</b> {churn_rate:.2f}%
    - Karyawan dengan <b>performa rendah</b> memiliki kemungkinan churn lebih tinggi.
    - Kelompok usia <b>21–35 tahun</b> menunjukkan tingkat churn tertinggi, menandakan potensi turnover awal karier.
    - Karyawan <b>menikah dan memiliki anak</b> cenderung lebih stabil dan loyal terhadap perusahaan.
    - Pendapatan berperan penting: semakin rendah gaji relatif terhadap rata-rata, semakin besar potensi churn.
    </div>
    """, unsafe_allow_html=True)

    # ===== VISUAL INSIGHTS =====
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 Churn Rate by Age Group")
        fig, ax = plt.subplots(figsize=(5, 3))
        churn_age.plot(kind="bar", color="#6C5CE7", ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Age Group")
        ax.set_title("Churn by Age", fontsize=12, weight="bold")
        for i, v in enumerate(churn_age.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
        st.pyplot(fig)

    with col2:
        st.markdown("#### 🧾 Churn Rate by Performance")
        fig, ax = plt.subplots(figsize=(5, 3))
        churn_perf.plot(kind="bar", color="#E67E22", ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Performance Score")
        ax.set_title("Churn by Performance", fontsize=12, weight="bold")
        for i, v in enumerate(churn_perf.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
        st.pyplot(fig)

    # ===== RECOMMENDATION SECTION =====
    st.markdown("<h2 style='color:#f8f9fa;'>🧭 HR Strategic Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:15px;color:#000000;'>
        <div style='background-color:#E3F2FD;padding:15px;border-radius:12px;border-left:5px solid #2196F3;'>
            <b>1️⃣ Retention Program for Young Employees</b><br>
            Bangun engagement melalui pelatihan karier, jalur promosi, dan mentoring untuk karyawan usia 21–35 tahun.
        </div>
        <div style='background-color:#FFF3E0;padding:15px;border-radius:12px;border-left:5px solid #FB8C00'>
            <b>2️⃣ Performance Improvement Plan</b><br>
            Terapkan coaching dan mentoring berkala bagi karyawan dengan performa rendah guna meningkatkan loyalitas.
        </div>
        <div style='background-color:#E8F5E9;padding:15px;border-radius:12px;border-left:5px solid #43A047'>
            <b>3️⃣ Review Compensation Policy</b><br>
            Evaluasi struktur gaji dan benefit untuk posisi dengan tingkat churn tinggi agar tetap kompetitif di pasar.
        </div>
        <div style='background-color:#F3E5F5;padding:15px;border-radius:12px;border-left:5px solid #8E24AA'>
            <b>4️⃣ Work-Life Balance Initiative</b><br>
            Berikan fleksibilitas kerja (hybrid, family support, wellness program) untuk meningkatkan retensi karyawan berkeluarga.
        </div>
        <div style='background-color:#FFFDE7;padding:15px;border-radius:12px;border-left:5px solid #FDD835'>
            <b>5️⃣ Early Warning System</b><br>
            Gunakan model prediksi churn untuk mendeteksi dini karyawan berisiko tinggi dan lakukan intervensi preventif.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== MODEL PERFORMANCE =====
    st.markdown(f"""
    <div style='margin-top:25px;background-color:#F0F4C3;padding:15px;border-radius:10px;border-left:5px solid #AFB42B;color:#000000;'>
    <b>Model Accuracy:</b> {accuracy*100:.2f}%
    Model ini dapat digunakan sebagai alat bantu <i>early detection</i> terhadap potensi churn, namun tetap perlu validasi periodik terhadap data aktual HR.
    </div>
    """, unsafe_allow_html=True)

    # ===== WATERMARK / SIGNATURE =====
    st.markdown("""
    <hr style='margin-top:40px;margin-bottom:10px'>
    <div style='text-align:center; color:gray; font-size:13px'>
    © 2025 — Employee Churn Dashboard by <b>Nauval Azhar</b> | Built using Streamlit
    </div>
    """, unsafe_allow_html=True)
