import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# =============================
# KONFIGURASI AWAL
# =============================
st.set_page_config(
    page_title="Career Analysis Dashboard",
    layout="wide"
)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_excel("career_dataset_large.xlsx")

df = load_data()

# =============================
# DATA CLEANING
# =============================
df["Education Level"] = df["Education Level"].str.strip()
df["Certifications"] = df["Certifications"].fillna("None")
df["Specialization"] = df["Specialization"].fillna("None")
df["Skills"] = df["Skills"].fillna("None")

df["Status_Keberhasilan"] = np.where(
    df["CGPA/Percentage"] >= 80,
    "Berhasil",
    "Gagal"
)

target_levels = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]
df = df[df["Education Level"].isin(target_levels)]

# =============================
# SIDEBAR MENU
# =============================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio(
    "Pilih Analisis",
    [
        "Ringkasan Data",
        "Analisis Profil Lengkap",
        "Analisis Skill",
        "Visualisasi Global"
    ]
)

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

    st.title("📈 Analisis Profil Lengkap (Berhasil vs Gagal)")

    selected_level = st.selectbox(
        "🎓 Pilih Jenjang Pendidikan",
        target_levels
    )

    def get_top_items(dataframe, edu_level, status, column, top_n=5):
        subset = dataframe[
            (dataframe["Education Level"] == edu_level) &
            (dataframe["Status_Keberhasilan"] == status)
        ][column]

        items = []
        for entry in subset:
            items.extend([i.strip() for i in str(entry).split(",")])

        return pd.DataFrame(
            Counter(items).most_common(top_n),
            columns=["Item", "Count"]
        )

    if selected_level in df["Education Level"].unique():

        fig, axes = plt.subplots(4, 2, figsize=(16, 22))
        fig.suptitle(
            f"Analisis Profil Lengkap: {selected_level}",
            fontsize=20,
            fontweight="bold"
        )

        # Specialization
        sns.barplot(
            data=get_top_items(df, selected_level, "Berhasil", "Specialization"),
            x="Count", y="Item", ax=axes[0, 0], palette="Greens_r"
        )
        axes[0, 0].set_title("Specialization (Berhasil)")

        sns.barplot(
            data=get_top_items(df, selected_level, "Gagal", "Specialization"),
            x="Count", y="Item", ax=axes[0, 1], palette="Reds_r"
        )
        axes[0, 1].set_title("Specialization (Gagal)")

        # Skills
        sns.barplot(
            data=get_top_items(df, selected_level, "Berhasil", "Skills"),
            x="Count", y="Item", ax=axes[1, 0], palette="Greens_r"
        )
        axes[1, 0].set_title("Skills (Berhasil)")

        sns.barplot(
            data=get_top_items(df, selected_level, "Gagal", "Skills"),
            x="Count", y="Item", ax=axes[1, 1], palette="Reds_r"
        )
        axes[1, 1].set_title("Skills (Gagal)")

        # Certifications
        sns.barplot(
            data=get_top_items(df, selected_level, "Berhasil", "Certifications"),
            x="Count", y="Item", ax=axes[2, 0], palette="Greens_r"
        )
        axes[2, 0].set_title("Certifications (Berhasil)")

        sns.barplot(
            data=get_top_items(df, selected_level, "Gagal", "Certifications"),
            x="Count", y="Item", ax=axes[2, 1], palette="Reds_r"
        )
        axes[2, 1].set_title("Certifications (Gagal)")

        # CGPA
        sns.histplot(
            df[(df["Education Level"] == selected_level) &
               (df["Status_Keberhasilan"] == "Berhasil")]["CGPA/Percentage"],
            ax=axes[3, 0], kde=True, color="green"
        )
        axes[3, 0].set_title("Distribusi CGPA (Berhasil)")

        sns.histplot(
            df[(df["Education Level"] == selected_level) &
               (df["Status_Keberhasilan"] == "Gagal")]["CGPA/Percentage"],
            ax=axes[3, 1], kde=True, color="red"
        )
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
    ).dropna().head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    diff_skill.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)

# =============================
# VISUALISASI GLOBAL
# =============================
elif menu == "Visualisasi Global":

    st.title("🌍 Visualisasi Global")

    tab1, tab2 = st.tabs(["Heatmap", "WordCloud"])

    with tab1:
        ct = pd.crosstab(df["Education Level"], df["Recommended Career"])
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(ct, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    with tab2:
        wc = WordCloud(
            width=900,
            height=400,
            background_color="white"
        ).generate(" ".join(df["Skills"].astype(str)))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
