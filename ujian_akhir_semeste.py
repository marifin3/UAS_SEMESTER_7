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
    st.error("File tidak ditemukan / gagal dibaca")
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
# TOP SKILL DIFFERENCE
# =============================
def extract_skills(series):
    items = []
    for s in series:
        items.extend([i.strip() for i in str(s).split(",")])
    return pd.Series(items).value_counts()

skill_ok = extract_skills(df[df["Status_Keberhasilan"] == "Berhasil"]["Skills"])
skill_fail = extract_skills(df[df["Status_Keberhasilan"] == "Gagal"]["Skills"])

diff_skill = (skill_ok - skill_fail).dropna().sort_values(ascending=False).head(10)

st.subheader("Top Skill Pembeda")

fig, ax = plt.subplots(figsize=(10, 6))
diff_skill.plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_xlabel("Selisih Frekuensi")
st.pyplot(fig)

# =============================
# BOXPLOT CGPA
# =============================
st.subheader("Sebaran CGPA per Karier")

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
st.subheader("Heatmap Spesialisasi vs Karier")

ct = pd.crosstab(df["Specialization"], df["Recommended Career"])

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(ct, cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# =============================
# STACKED BAR
# =============================
st.subheader("Distribusi Karier per Jenjang (%)")

career_dist = (
    pd.crosstab(df["Education Level"], df["Recommended Career"], normalize="index") * 100
)

fig, ax = plt.subplots(figsize=(12, 6))
career_dist.plot(kind="bar", stacked=True, ax=ax)
st.pyplot(fig)

# =============================
# WORDCLOUD
# =============================
st.subheader("WordCloud Skills")

all_skills = " ".join(df["Skills"].astype(str))

wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis"
).generate(all_skills)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc)
ax.axis("off")
st.pyplot(fig)

