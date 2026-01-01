import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

st.set_page_config(page_title="Career Analysis", layout="wide")

# =============================
# LOAD DATA
# =============================
FILE_PATH = "career_dataset_large.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH, engine="openpyxl")

try:
    df = load_data()
    st.success("Data berhasil dimuat")
except Exception as e:
    st.error(f"File tidak ditemukan / gagal dibaca: {e}")
    st.stop()

# =============================
# DATA CLEANING
# =============================
df["Education Level"] = df["Education Level"].str.strip()
df["Certifications"] = df["Certifications"].fillna("None")
df["Specialization"] = df["Specialization"].fillna("None")
df["Skills"] = df["Skills"].fillna("None")

# Feature Engineering sesuai kriteria laporan
df["Status_Keberhasilan"] = np.where(
    df["CGPA/Percentage"] >= 80, "Berhasil", "Gagal"
)

target_levels = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]
df = df[df["Education Level"].isin(target_levels)]

# =============================
# SUMMARY TABLE
# =============================
st.subheader("Ringkasan Jenjang Pendidikan")

summary = (
    df["Education Level"]
    .value_counts()
    .reindex(target_levels)
    .reset_index()
)
summary.columns = ["Education Level", "Total Data"]

st.dataframe(summary, use_container_width=True)

# =============================
# ANALISIS PROFIL LENGKAP
# =============================
st.header("🔍 Analisis Profil Lengkap per Jenjang Pendidikan")
st.caption("Perbandingan karakteristik individu **Berhasil** dan **Gagal**")

def get_top_items(dataframe, edu_level, status, column, top_n=3):
    subset = dataframe[
        (dataframe["Education Level"] == edu_level) & 
        (dataframe["Status_Keberhasilan"] == status)
    ][column].dropna()

    items = []
    for entry in subset:
        parts = [i.strip() for i in str(entry).split(",")]
        for p in parts:
            if p.lower() not in ["none", "nan", ""] and p is not None:
                items.append(p)

    counts = Counter(items).most_common(top_n)
    return pd.DataFrame(counts, columns=["Item", "Count"])

for jenjang in target_levels:
    if jenjang not in df["Education Level"].unique():
        continue

    with st.expander(f"🎓 {jenjang}"):
        fig, axes = plt.subplots(4, 2, figsize=(16, 18))
        fig.suptitle(
            f"Profil Lengkap: {jenjang} (Berhasil vs Gagal)",
            fontsize=18,
            fontweight="bold"
        )

        # 1. Specialization
        sns.barplot(
            data=get_top_items(df, jenjang, "Berhasil", "Specialization"),
            x="Count", y="Item", ax=axes[0, 0], palette="Greens_r"
        )
        axes[0, 0].set_title("Top 3 Specialization (Berhasil)")

        sns.barplot(
            data=get_top_items(df, jenjang, "Gagal", "Specialization"),
            x="Count", y="Item", ax=axes[0, 1], palette="Reds_r"
        )
        axes[0, 1].set_title("Top 3 Specialization (Gagal)")

        # 2. Skills
        sns.barplot(
            data=get_top_items(df, jenjang, "Berhasil", "Skills"),
            x="Count", y="Item", ax=axes[1, 0], palette="Greens_r"
        )
        axes[1, 0].set_title("Top 3 Skills (Berhasil)")

        sns.barplot(
            data=get_top_items(df, jenjang, "Gagal", "Skills"),
            x="Count", y="Item", ax=axes[1, 1], palette="Reds_r"
        )
        axes[1, 1].set_title("Top 3 Skills (Gagal)")

        # 3. Certifications
        sns.barplot(
            data=get_top_items(df, jenjang, "Berhasil", "Certifications"),
            x="Count", y="Item", ax=axes[2, 0], palette="Greens_r"
        )
        axes[2, 0].set_title("Top 3 Certifications (Berhasil)")

        sns.barplot(
            data=get_top_items(df, jenjang, "Gagal", "Certifications"),
            x="Count", y="Item", ax=axes[2, 1], palette="Reds_r"
        )
        axes[2, 1].set_title("Top 3 Certifications (Gagal)")

        # 4. Distribusi CGPA
        sns.histplot(
            df[(df["Education Level"] == jenjang) & 
               (df["Status_Keberhasilan"] == "Berhasil")]["CGPA/Percentage"],
            kde=True, ax=axes[3, 0], color="green"
        )
        axes[3, 0].set_title("Distribusi CGPA (Berhasil)")

        sns.histplot(
            df[(df["Education Level"] == jenjang) & 
               (df["Status_Keberhasilan"] == "Gagal")]["CGPA/Percentage"],
            kde=True, ax=axes[3, 1], color="red"
        )
        axes[3, 1].set_title("Distribusi CGPA (Gagal)")

        plt.tight_layout()
        st.pyplot(fig)

st.divider()

# =============================
# ANALISIS NILAI TERPOPULER
# =============================
st.subheader("📊 Analisis Nilai Terpopuler per Jenjang Pendidikan")

cols = ['Specialization', 'Skills', 'Certifications', 'CGPA/Percentage', 'Recommended Career']
results = []

for level in target_levels:
    level_df = df[df['Education Level'] == level]
    if level_df.empty: continue

    for col in cols:
        if col not in level_df.columns: continue
        counts = level_df[col].value_counts(dropna=True)
        if counts.empty: continue

        results.append({
            'Jenjang': level,
            'Kolom': col,
            'Jumlah Unik (Beda)': int(level_df[col].nunique()),
            'Nilai Terpopuler': counts.index[0],
            'Frekuensi (Sama)': int(counts.iloc[0])
        })

summary_df = pd.DataFrame(results)
if summary_df.empty:
    st.warning("Tidak ada data yang dapat ditampilkan.")
else:
    st.dataframe(summary_df, use_container_width=True)

# =============================
# FIX: HITUNG PERSENTASE KEBERHASILAN (Solusi NameError)
# =============================
# Menghitung tabel silang dan persentase keberhasilan sebelum dipanggil
success_counts = pd.crosstab(df["Education Level"], df["Status_Keberhasilan"])
success_pct = success_counts.div(success_counts.sum(axis=1), axis=0) * 100
success_pct = success_pct.reindex(target_levels)

# =============================
# HEATMAP: SPESIALISASI vs KARIER
# =============================
st.subheader("🔥 Heatmap Spesialisasi vs Rekomendasi Karier")

with st.expander("📊 Lihat Heatmap Korelasi", expanded=True):
    selected_level = st.selectbox(
        "🎓 Filter Jenjang Pendidikan (Opsional)",
        options=["Semua"] + target_levels
    )

    heat_df = df if selected_level == "Semua" else df[df["Education Level"] == selected_level]

    if heat_df.empty:
        st.warning("⚠️ Data tidak tersedia.")
    else:
        ct = pd.crosstab(heat_df["Specialization"], heat_df["Recommended Career"])
        if not ct.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, ax=ax)
            ax.set_title("Heatmap: Korelasi Spesialisasi vs Rekomendasi Karier", fontsize=16, fontweight="bold")
            st.pyplot(fig)

# -----------------------------
# TABEL RINGKASAN (Sekarang success_pct sudah ada)
# -----------------------------
st.markdown("### 📋 Tabel Persentase Keberhasilan")
st.dataframe(success_pct.round(2), use_container_width=True)

# =============================
# WORDCLOUD
# =============================
st.subheader("WordCloud Skills")
all_skills = " ".join(df["Skills"].astype(str))
wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(all_skills)
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig)
