import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Career Path Analysis",
    layout="wide"
)

# =============================
# HEADER
# =============================
st.title("🎓 Career Path Analysis Dashboard")
st.markdown(
    """
    Dashboard ini menyajikan **analisis hubungan antara jenjang pendidikan,
    keterampilan, sertifikasi, dan keberhasilan karier** berdasarkan data historis.

    📌 Fokus Analisis:
    - Perbandingan **Berhasil vs Gagal**
    - Profil kompetensi tiap jenjang pendidikan
    - Distribusi karier dan performa akademik
    """
)
st.divider()

# =============================
# LOAD DATA
# =============================
FILE_PATH = "career_dataset_large.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH, engine="openpyxl")

try:
    df = load_data()
    st.success("✅ Data berhasil dimuat")
except Exception:
    st.error("❌ File tidak ditemukan / gagal dibaca")
    st.stop()

# =============================
# DATA CLEANING
# =============================
df["Education Level"] = df["Education Level"].str.strip()
df["Certifications"] = df["Certifications"].fillna("None")
df["Specialization"] = df["Specialization"].fillna("None")
df["Skills"] = df["Skills"].fillna("None")

df["Status_Keberhasilan"] = np.where(
    df["CGPA/Percentage"] >= 80, "Berhasil", "Gagal"
)

target_levels = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]
df = df[df["Education Level"].isin(target_levels)]

# =============================
# SUMMARY TABLE
# =============================
st.header("📋 Ringkasan Distribusi Data")

summary = (
    df["Education Level"]
    .value_counts()
    .reindex(target_levels)
    .reset_index()
)
summary.columns = ["Education Level", "Total Data"]

col1, col2 = st.columns([2, 3])
with col1:
    st.dataframe(summary, use_container_width=True)

with col2:
    st.info(
        """
        **Interpretasi:**
        Tabel ini menunjukkan jumlah data pada masing-masing jenjang pendidikan
        yang menjadi dasar analisis lanjutan.
        """
    )

st.divider()

# =============================
# ANALISIS PROFIL LENGKAP
# =============================
st.header("🔍 Analisis Profil Lengkap per Jenjang Pendidikan")
st.caption("Perbandingan karakteristik individu **Berhasil** dan **Gagal**")

def get_top_items(dataframe, edu_level, status, column, top_n=5):
    subset = dataframe[
        (dataframe["Education Level"] == edu_level) &
        (dataframe["Status_Keberhasilan"] == status)
    ][column].dropna()

    items = []
    for entry in subset:
        items.extend([i.strip() for i in str(entry).split(",")])

    counts = Counter(items).most_common(top_n)
    return pd.DataFrame(counts, columns=["Item", "Count"])

for jenjang in target_levels:
    if jenjang not in df["Education Level"].unique():
        continue

    with st.expander(f"🎓 {jenjang}"):
        fig, axes = plt.subplots(4, 2, figsize=(16, 22))
        fig.suptitle(
            f"Profil Lengkap: {jenjang} (Berhasil vs Gagal)",
            fontsize=18,
            fontweight="bold"
        )

        # Specialization
        sns.barplot(
            data=get_top_items(df, jenjang, "Berhasil", "Specialization"),
            x="Count", y="Item", ax=axes[0, 0], palette="Greens_r"
        )
        axes[0, 0].set_title("Top Specialization (Berhasil)")

        sns.barplot(
            data=get_top_items(df, jenjang, "Gagal", "Specialization"),
            x="Count", y="Item", ax=axes[0, 1], palette="Reds_r"
        )
        axes[0, 1].set_title("Top Specialization (Gagal)")

        # Skills
        sns.barplot(
            data=get_top_items(df, jenjang, "Berhasil", "Skills"),
            x="Count", y="Item", ax=axes[1, 0], palette="Greens_r"
        )
        axes[1, 0].set_title("Top Skills (Berhasil)")

        sns.barplot(
            data=get_top_items(df, jenjang, "Gagal", "Skills"),
            x="Count", y="Item", ax=axes[1, 1], palette="Reds_r"
        )
        axes[1, 1].set_title("Top Skills (Gagal)")

        # Certifications
        sns.barplot(
            data=get_top_items(df, jenjang, "Berhasil", "Certifications"),
            x="Count", y="Item", ax=axes[2, 0], palette="Greens_r"
        )
        axes[2, 0].set_title("Top Certifications (Berhasil)")

        sns.barplot(
            data=get_top_items(df, jenjang, "Gagal", "Certifications"),
            x="Count", y="Item", ax=axes[2, 1], palette="Reds_r"
        )
        axes[2, 1].set_title("Top Certifications (Gagal)")

        # CGPA
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
st.header("📊 Nilai Terpopuler per Jenjang")
st.caption("Ringkasan nilai yang paling sering muncul")

cols = ['Specialization', 'Skills', 'Certifications', 'CGPA/Percentage', 'Recommended Career']
results = []

for level in target_levels:
    level_df = df[df["Education Level"] == level]
    for col in cols:
        if col in level_df.columns:
            counts = level_df[col].value_counts()
            if not counts.empty:
                results.append({
                    "Jenjang": level,
                    "Kolom": col,
                    "Jumlah Unik": level_df[col].nunique(),
                    "Nilai Terpopuler": counts.index[0],
                    "Frekuensi": counts.iloc[0]
                })

summary_df = pd.DataFrame(results)

with st.expander("📌 Lihat Tabel Nilai Terpopuler"):
    st.dataframe(summary_df, use_container_width=True)

st.divider()

# =============================
# TOP SKILL DIFFERENCE
# =============================
st.header("🧠 Top Skill Pembeda Keberhasilan")

def extract_skills(series):
    items = []
    for s in series:
        items.extend([i.strip() for i in str(s).split(",")])
    return pd.Series(items).value_counts()

diff_skill = (
    extract_skills(df[df["Status_Keberhasilan"] == "Berhasil"]["Skills"]) -
    extract_skills(df[df["Status_Keberhasilan"] == "Gagal"]["Skills"])
).dropna().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
diff_skill.plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_xlabel("Selisih Frekuensi")
st.pyplot(fig)

st.info("Skill dengan nilai positif lebih dominan dimiliki individu yang berhasil.")

st.divider()

# =============================
# WORDCLOUD
# =============================
st.header("☁️ WordCloud Keterampilan Dominan")

all_skills = " ".join(df["Skills"].astype(str))
wc = WordCloud(width=900, height=400, background_color="white").generate(all_skills)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig)
