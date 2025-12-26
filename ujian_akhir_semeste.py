import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

st.set_page_config(page_title="Career Analysis Dashboard", layout="wide")

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("📌 Navigasi Menu")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Ringkasan Data",
        "Analisis Profil Lengkap",
        "Nilai Terpopuler",
        "Skill Pembeda",
        "Visualisasi Karier",
        "Heatmap",
        "WordCloud"
    ]
)

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
# RINGKASAN DATA
# =============================
if menu == "Ringkasan Data":

    st.title("📊 Ringkasan Jenjang Pendidikan")

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
elif menu == "Analisis Profil Lengkap":

    st.title("🔍 Analisis Profil Lengkap")
    st.caption("Perbandingan Berhasil vs Gagal per Jenjang")

    def get_top_items(dataframe, edu_level, status, column, top_n=5):
        subset = dataframe[
            (dataframe["Education Level"] == edu_level) &
            (dataframe["Status_Keberhasilan"] == status)
        ][column].dropna()

        items = []
        for entry in subset:
            items.extend([i.strip() for i in str(entry).split(",")])

        return pd.DataFrame(
            Counter(items).most_common(top_n),
            columns=["Item", "Count"]
        )

    for jenjang in target_levels:
        with st.expander(f"🎓 {jenjang}", expanded=False):

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            sns.barplot(
                data=get_top_items(df, jenjang, "Berhasil", "Skills"),
                x="Count", y="Item", ax=axes[0, 0], palette="Greens_r"
            )
            axes[0, 0].set_title("Top Skills (Berhasil)")

            sns.barplot(
                data=get_top_items(df, jenjang, "Gagal", "Skills"),
                x="Count", y="Item", ax=axes[0, 1], palette="Reds_r"
            )
            axes[0, 1].set_title("Top Skills (Gagal)")

            sns.histplot(
                df[(df["Education Level"] == jenjang) &
                   (df["Status_Keberhasilan"] == "Berhasil")]["CGPA/Percentage"],
                kde=True, ax=axes[1, 0], color="green"
            )
            axes[1, 0].set_title("CGPA (Berhasil)")

            sns.histplot(
                df[(df["Education Level"] == jenjang) &
                   (df["Status_Keberhasilan"] == "Gagal")]["CGPA/Percentage"],
                kde=True, ax=axes[1, 1], color="red"
            )
            axes[1, 1].set_title("CGPA (Gagal)")

            st.pyplot(fig)

# =============================
# NILAI TERPOPULER
# =============================
elif menu == "Nilai Terpopuler":

    st.title("📊 Nilai Terpopuler per Jenjang")

    cols = [
        'Specialization',
        'Skills',
        'Certifications',
        'Recommended Career'
    ]

    results = []

    for level in target_levels:
        level_df = df[df['Education Level'] == level]
        for col in cols:
            counts = level_df[col].value_counts()
            if not counts.empty:
                results.append({
                    "Jenjang": level,
                    "Kolom": col,
                    "Nilai Terpopuler": counts.index[0],
                    "Frekuensi": counts.iloc[0]
                })

    st.dataframe(pd.DataFrame(results), use_container_width=True)

# =============================
# SKILL PEMBEDA
# =============================
elif menu == "Skill Pembeda":

    st.title("🧠 Skill Pembeda Berhasil vs Gagal")

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
    st.pyplot(fig)

# =============================
# VISUALISASI KARIER
# =============================
elif menu == "Visualisasi Karier":

    st.title("📦 Distribusi Karier")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="Recommended Career",
        y="CGPA/Percentage",
        ax=ax
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =============================
# HEATMAP
# =============================
elif menu == "Heatmap":

    st.title("🔥 Heatmap Spesialisasi vs Karier")

    ct = pd.crosstab(df["Specialization"], df["Recommended Career"])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(ct, annot=False, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# =============================
# WORDCLOUD
# =============================
elif menu == "WordCloud":

    st.title("☁️ WordCloud Skills")

    all_skills = " ".join(df["Skills"].astype(str))

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(all_skills)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)
