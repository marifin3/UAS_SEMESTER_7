
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

st.subheader("📊 Ringkasan Jenjang Pendidikan")

# =============================
# 1. MEMBACA FILE EXCEL
# =============================
FILE_PATH = "career_dataset_large.xlsx"

@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH, engine="openpyxl")

try:
    df = load_data()
    st.success("File Excel berhasil dimuat")
except Exception:
    st.error("File tidak ditemukan. Pastikan 'career_dataset_large.xlsx' ada di repository.")
    st.stop()

# =============================
# 2. PEMBERSIHAN DATA
# =============================
df["Education Level"] = df["Education Level"].str.strip()

# =============================
# 3. FILTER DATA
# =============================
target_levels = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]
df_filtered = df[df["Education Level"].isin(target_levels)].copy()

# =============================
# 4. TABEL RINGKASAN
# =============================
summary_table = (
    df_filtered["Education Level"]
    .value_counts()
    .reindex(target_levels)
    .reset_index()
)
summary_table.columns = ["Education Level", "Total Data"]

# =============================
# 5. TAMPILKAN HASIL
# =============================
st.markdown("### 📌 Tabel Jenjang Pendidikan (Sesuai Filter)")
st.dataframe(summary_table, use_container_width=True)

st.markdown("### 📄 Contoh Data Setelah Filter (20 baris pertama)")
st.dataframe(df_filtered.head(20), use_container_width=True)


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
# 2. PENGOLAHAN DATA
# =============================
df['Education Level'] = df['Education Level'].str.strip()

df['Status_Keberhasilan'] = np.where(
    df['CGPA/Percentage'] >= 80, 'Berhasil', 'Gagal'
)

df['Certifications'] = df['Certifications'].fillna('None')
df['Specialization'] = df['Specialization'].fillna('None')
df['Skills'] = df['Skills'].fillna('None')


# =============================
# 3. FUNGSI TOP ITEMS
# =============================
def get_top_items(dataframe, edu_level, status, column, top_n=5):
    subset = dataframe[
        (dataframe['Education Level'] == edu_level) &
        (dataframe['Status_Keberhasilan'] == status)
    ][column].dropna()

    items = []
    for entry in subset:
        items.extend([i.strip() for i in str(entry).split(',')])

    counts = Counter(items).most_common(top_n)
    return pd.DataFrame(counts, columns=['Item', 'Count'])


# =============================
# 4. VISUALISASI PER JENJANG
# =============================
list_jenjang = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]

st.header("Analisis Profil Lengkap per Jenjang Pendidikan")

for jenjang in list_jenjang:
    if jenjang not in df['Education Level'].unique():
        st.warning(f"Data untuk jenjang {jenjang} tidak ditemukan")
        continue

    st.subheader(f"{jenjang} — Berhasil vs Gagal")

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle(
        f'Analisis Profil Lengkap: {jenjang}',
        fontsize=22,
        fontweight='bold',
        y=1.02
    )

    # ---- Specialization ----
    d_sp_b = get_top_items(df, jenjang, 'Berhasil', 'Specialization')
    d_sp_g = get_top_items(df, jenjang, 'Gagal', 'Specialization')

    if not d_sp_b.empty:
        sns.barplot(data=d_sp_b, x='Count', y='Item', ax=axes[0, 0], palette='Greens_r')
    axes[0, 0].set_title('Top Specialization (Berhasil)')

    if not d_sp_g.empty:
        sns.barplot(data=d_sp_g, x='Count', y='Item', ax=axes[0, 1], palette='Reds_r')
    axes[0, 1].set_title('Top Specialization (Gagal)')

    # ---- Skills ----
    d_s_b = get_top_items(df, jenjang, 'Berhasil', 'Skills')
    d_s_g = get_top_items(df, jenjang, 'Gagal', 'Skills')

    if not d_s_b.empty:
        sns.barplot(data=d_s_b, x='Count', y='Item', ax=axes[1, 0], palette='Greens_r')
    axes[1, 0].set_title('Top Skills (Berhasil)')

    if not d_s_g.empty:
        sns.barplot(data=d_s_g, x='Count', y='Item', ax=axes[1, 1], palette='Reds_r')
    axes[1, 1].set_title('Top Skills (Gagal)')

    # ---- Certifications ----
    d_c_b = get_top_items(df, jenjang, 'Berhasil', 'Certifications')
    d_c_g = get_top_items(df, jenjang, 'Gagal', 'Certifications')

    if not d_c_b.empty:
        sns.barplot(data=d_c_b, x='Count', y='Item', ax=axes[2, 0], palette='Greens_r')
    axes[2, 0].set_title('Top Certifications (Berhasil)')

    if not d_c_g.empty:
        sns.barplot(data=d_c_g, x='Count', y='Item', ax=axes[2, 1], palette='Reds_r')
    axes[2, 1].set_title('Top Certifications (Gagal)')

    # ---- CGPA Distribution ----
    sub_b = df[(df['Education Level'] == jenjang) & (df['Status_Keberhasilan'] == 'Berhasil')]
    sub_g = df[(df['Education Level'] == jenjang) & (df['Status_Keberhasilan'] == 'Gagal')]

    sns.histplot(sub_b['CGPA/Percentage'], kde=True, ax=axes[3, 0], color='green')
    axes[3, 0].set_title('Distribusi CGPA (Berhasil)')

    sns.histplot(sub_g['CGPA/Percentage'], kde=True, ax=axes[3, 1], color='red')
    axes[3, 1].set_title('Distribusi CGPA (Gagal)')

    plt.tight_layout()
    st.pyplot(fig)

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

