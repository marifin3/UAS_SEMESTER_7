import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Analisis Karier", layout="wide")

st.title("🎓 Dashboard Analisis Pendidikan & Karier")

# --- 2. FUNGSI LOAD DATA ---
file_path = 'career_dataset_large.xlsx'

@st.cache_data
def load_data():
    try:
        # Menambahkan engine='openpyxl' sangat penting untuk file .xlsx
        df = pd.read_excel(file_path, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error: Tidak dapat menemukan file '{file_path}'.")
        return None

df = load_data()

# --- 3. PROSES UTAMA (Hanya jalan jika data ada) ---
if df is not None:
    # Pembersihan Data Awal
    df['Education Level'] = df['Education Level'].str.strip()
    df['Status_Keberhasilan'] = np.where(df['CGPA/Percentage'] >= 80, 'Berhasil', 'Gagal')
    df['Certifications'] = df['Certifications'].fillna('None')
    df['Specialization'] = df['Specialization'].fillna('None')
    df['Skills'] = df['Skills'].fillna('None')

    # Menu Navigasi Sidebar
    st.sidebar.title("Menu Analisis")
    menu = st.sidebar.radio("Pilih Visualisasi:", 
                            ["Ringkasan Data", "Analisis Skill & Karier", "Profil Detail Per Jenjang"])

    # --- HALAMAN 1: RINGKASAN DATA ---
    if menu == "Ringkasan Data":
        st.header("📊 Ringkasan Distribusi & Keberhasilan")
        
        target_levels = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]
        df_filtered = df[df['Education Level'].isin(target_levels)].copy()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Proporsi Data per Jenjang")
            dist_counts = df_filtered['Education Level'].value_counts(normalize=True) * 100
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(dist_counts, labels=dist_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
            st.pyplot(fig_pie)

        with col2:
            st.subheader("Persentase Berhasil vs Gagal")
            success_pct = pd.crosstab(df_filtered['Education Level'], df_filtered['Status_Keberhasilan'], normalize='index') * 100
            fig_bar, ax_bar = plt.subplots()
            success_pct.reindex(target_levels).plot(kind='bar', stacked=True, ax=ax_bar, color=['#2ca02c', '#d62728'])
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

        st.divider()
        st.subheader("📄 Contoh 10 Data Mentah")
        st.dataframe(df_filtered.head(10))

    # --- HALAMAN 2: ANALISIS SKILL & KARIER ---
    elif menu == "Analisis Skill & Karier":
        st.header("💡 Analisis Skill, Karier & Nilai")

        # Visualisasi Boxplot
        st.subheader("Sebaran CGPA Berdasarkan Karier")
        fig_box, ax_box = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='Recommended Career', y='CGPA/Percentage', palette='Set3', ax=ax_box)
        plt.xticks(rotation=45)
        st.pyplot(fig_box)

        # WordCloud
        st.divider()
        st.subheader("☁️ Word Cloud: Skill Paling Populer")
        all_skills = " ".join(df['Skills'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_skills)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        # Heatmap
        st.divider()
        st.subheader("🔥 Heatmap: Spesialisasi vs Karier")
        fig_heat, ax_heat = plt.subplots(figsize=(12, 8))
        ct = pd.crosstab(df['Specialization'], df['Recommended Career'])
        sns.heatmap(ct, annot=True, cmap='YlGnBu', fmt='d', ax=ax_heat)
        st.pyplot(fig_heat)

    # --- HALAMAN 3: PROFIL DETAIL PER JENJANG ---
    elif menu == "Profil Detail Per Jenjang":
        st.header("🔎 Analisis Mendalam Per Jenjang Pendidikan")
        
        jenjang_pilihan = st.selectbox("Pilih Jenjang:", ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"])
        
        def get_top_items(dataframe, edu_level, status, column):
            subset = dataframe[(dataframe['Education Level'] == edu_level) & (dataframe['Status_Keberhasilan'] == status)][column]
            all_items = []
            for entry in subset:
                items = [item.strip() for item in str(entry).split(',')]
                all_items.extend(items)
            return pd.DataFrame(Counter(all_items).most_common(5), columns=['Item', 'Count'])

        # Menampilkan perbandingan Berhasil vs Gagal
        col_b, col_g = st.columns(2)

        with col_b:
            st.success(f"Profil Kunci: {jenjang_pilihan} - BERHASIL")
            top_skills_b = get_top_items(df, jenjang_pilihan, 'Berhasil', 'Skills')
            fig_b, ax_b = plt.subplots()
            sns.barplot(data=top_skills_b, x='Count', y='Item', palette='Greens_r', ax=ax_b)
            st.pyplot(fig_b)

        with col_g:
            st.error(f"Profil Kunci: {jenjang_pilihan} - GAGAL")
            top_skills_g = get_top_items(df, jenjang_pilihan, 'Gagal', 'Skills')
            fig_g, ax_g = plt.subplots()
            sns.barplot(data=top_skills_g, x='Count', y='Item', palette='Reds_r', ax=ax_g)
            st.pyplot(fig_g)

else:
    st.warning("Pastikan file 'career_dataset_large.xlsx' sudah diunggah ke GitHub.")
