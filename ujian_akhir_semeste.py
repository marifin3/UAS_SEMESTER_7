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

# -----------------------------
# VISUALISASI
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# --- PIE CHART ---
colors = sns.color_palette("pastel")[0:5]
ax1.pie(
    dist_df["Percentage"],
    labels=dist_df["Education Level"],
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    explode=[0.05] * len(dist_df)
)
ax1.set_title("Proporsi Data per Jenjang Pendidikan", fontsize=14, fontweight="bold")

# --- STACKED BAR ---
success_pct.plot(
    kind="bar",
    stacked=True,
    ax=ax2,
    color=["#2ca02c", "#d62728"]
)

ax2.set_title("Persentase Berhasil vs Gagal per Jenjang", fontsize=14, fontweight="bold")
ax2.set_ylabel("Persentase (%)")
ax2.set_xlabel("Jenjang Pendidikan")
ax2.legend(title="Status", bbox_to_anchor=(1.05, 1))

# Label persentase
for p in ax2.patches:
    height = p.get_height()
    if height > 0:
        ax2.text(
            p.get_x() + p.get_width() / 2,
            p.get_y() + height / 2,
            f"{height:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

plt.tight_layout()
st.pyplot(fig)
