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
# SIDEBAR
# =============================
st.sidebar.title("🎓 Career Analysis")
menu = st.sidebar.radio(
    "Navigasi",
    ["Dashboard Utama", "Profil Lengkap", "Analisis Skill", "Visualisasi Global"]
)

st.sidebar.markdown("---")
st.sidebar.caption("© Career Analysis Dashboard")

# =============================
# LOAD DATA
# =============================
FILE_PATH = "career_dataset_large.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH, engine="openpyxl")

df = load_data()

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
# DASHBOARD UTAMA
# =============================
if menu == "Dashboard Utama":

    st.title("📊 Career Path Analysis Dashboard")
    st.caption("Ringkasan cepat data & performa karier")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Tingkat Berhasil (%)",
                round((df["Status_Keberhasilan"] == "Berhasil").mean() * 100, 2))
    col3.metric("Jumlah Karier Unik", df["Recommended Career"].nunique())

    st.markdown("---")

    st.subheader("📋 Distribusi Jenjang Pendidikan")
    summary = df["Education Level"].value_counts().reindex(target_levels)

    fig, ax = plt.subplots(figsize=(10, 4))
    summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("Jumlah Data")
    st.pyplot(fig)

# =============================
# ANALISIS PROFIL LENGKAP (BERHASIL vs GAGAL)
# =============================
st.subheader("📈 Analisis Profil Lengkap per Jenjang Pendidikan")

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

    st.markdown(f"### 🎓 {jenjang}")

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle(
        f"Analisis Profil Lengkap: {jenjang} (Berhasil vs Gagal)",
        fontsize=20,
        fontweight="bold",
        y=1.02
    )

    # ===== Specialization =====
    sp_b = get_top_items(df, jenjang, "Berhasil", "Specialization")
    sp_g = get_top_items(df, jenjang, "Gagal", "Specialization")

    if not sp_b.empty:
        sns.barplot(data=sp_b, x="Count", y="Item", ax=axes[0, 0], palette="Greens_r")
    axes[0, 0].set_title("Top Specialization (Berhasil)")

    if not sp_g.empty:
        sns.barplot(data=sp_g, x="Count", y="Item", ax=axes[0, 1], palette="Reds_r")
    axes[0, 1].set_title("Top Specialization (Gagal)")

    # ===== Skills =====
    sk_b = get_top_items(df, jenjang, "Berhasil", "Skills")
    sk_g = get_top_items(df, jenjang, "Gagal", "Skills")

    if not sk_b.empty:
        sns.barplot(data=sk_b, x="Count", y="Item", ax=axes[1, 0], palette="Greens_r")
    axes[1, 0].set_title("Top Skills (Berhasil)")

    if not sk_g.empty:
        sns.barplot(data=sk_g, x="Count", y="Item", ax=axes[1, 1], palette="Reds_r")
    axes[1, 1].set_title("Top Skills (Gagal)")

    # ===== Certifications =====
    ct_b = get_top_items(df, jenjang, "Berhasil", "Certifications")
    ct_g = get_top_items(df, jenjang, "Gagal", "Certifications")

    if not ct_b.empty:
        sns.barplot(data=ct_b, x="Count", y="Item", ax=axes[2, 0], palette="Greens_r")
    axes[2, 0].set_title("Top Certifications (Berhasil)")

    if not ct_g.empty:
        sns.barplot(data=ct_g, x="Count", y="Item", ax=axes[2, 1], palette="Reds_r")
    axes[2, 1].set_title("Top Certifications (Gagal)")

    # ===== CGPA Distribution =====
    sub_b = df[
        (df["Education Level"] == jenjang) &
        (df["Status_Keberhasilan"] == "Berhasil")
    ]

    sub_g = df[
        (df["Education Level"] == jenjang) &
        (df["Status_Keberhasilan"] == "Gagal")
    ]

    sns.histplot(sub_b["CGPA/Percentage"], kde=True, ax=axes[3, 0], color="green")
    axes[3, 0].set_title("Distribusi CGPA (Berhasil)")

    sns.histplot(sub_g["CGPA/Percentage"], kde=True, ax=axes[3, 1], color="red")
    axes[3, 1].set_title("Distribusi CGPA (Gagal)")

    plt.tight_layout()
    st.pyplot(fig)

# =============================
# ANALISIS SKILL
# =============================
elif menu == "Analisis Skill":

    st.title("🧠 Analisis Skill Pembeda")

    def extract_skills(series):
        items = []
        for s in series:
            items.extend([i.strip() for i in str(s).split(",")])
        return pd.Series(items).value_counts()

    diff_skill = (
        extract_skills(df[df["Status_Keberhasilan"] == "Berhasil"]["Skills"]) -
        extract_skills(df[df["Status_Keberhasilan"] == "Gagal"]["Skills"])
    ).dropna().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    diff_skill.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    st.pyplot(fig)

    st.info("Skill dengan nilai positif lebih dominan pada individu yang berhasil.")

# =============================
# VISUALISASI GLOBAL
# =============================
elif menu == "Visualisasi Global":

    st.title("🌍 Visualisasi Global")

    tab1, tab2 = st.tabs(["📦 Distribusi Karier", "☁️ WordCloud"])

    with tab1:
        ct = pd.crosstab(df["Education Level"], df["Recommended Career"])
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(ct, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    with tab2:
        all_skills = " ".join(df["Skills"].astype(str))
        wc = WordCloud(
            width=900,
            height=400,
            background_color="white"
        ).generate(all_skills)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
