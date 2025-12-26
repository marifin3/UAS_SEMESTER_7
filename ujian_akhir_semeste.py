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
# ANALISIS NILAI TERPOPULER (STREAMLIT)
# =============================
import streamlit as st
import pandas as pd

st.subheader("📊 Analisis Nilai Terpopuler per Jenjang Pendidikan")

# Kolom yang ingin dianalisis
cols = [
    'Specialization',
    'Skills',
    'Certifications',
    'CGPA/Percentage',
    'Recommended Career'
]
target_levels = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]

results = []

for level in target_levels:
    level_df = df[df['Education Level'] == level]

    # Lewati jika tidak ada data
    if level_df.empty:
        continue

    for col in cols:
        # Pastikan kolom ada
        if col not in level_df.columns:
            continue

        counts = level_df[col].value_counts(dropna=True)

        if counts.empty:
            continue

        results.append({
            'Jenjang': level,
            'Kolom': col,
            'Jumlah Unik (Beda)': int(level_df[col].nunique()),
            'Nilai Terpopuler': counts.index[0],
            'Frekuensi (Sama)': int(counts.iloc[0])
        })

# Konversi ke DataFrame
summary_df = pd.DataFrame(results)

# Tampilkan di Streamlit
if summary_df.empty:
    st.warning("Tidak ada data yang dapat ditampilkan.")
else:
    st.dataframe(summary_df, use_container_width=True)


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

# -----------------------------
# PERHITUNGAN PERSENTASE
# -----------------------------
# A. Distribusi keseluruhan (Pie)
dist_counts = df["Education Level"].value_counts(normalize=True) * 100
dist_df = dist_counts.reindex(target_levels).reset_index()
dist_df.columns = ["Education Level", "Percentage"]

# B. Persentase Berhasil vs Gagal
success_counts = (
    df.groupby(["Education Level", "Status_Keberhasilan"])
      .size()
      .unstack(fill_value=0)
)

success_pct = (
    success_counts
    .div(success_counts.sum(axis=1), axis=0) * 100
).reindex(target_levels)

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

# =============================
# HEATMAP: SPESIALISASI vs KARIER (STREAMLIT SUPPORT)
# =============================
st.subheader("🔥 Heatmap Spesialisasi vs Rekomendasi Karier")

with st.expander("📊 Lihat Heatmap Korelasi", expanded=True):

    # Filter opsional berdasarkan jenjang
    selected_level = st.selectbox(
        "🎓 Filter Jenjang Pendidikan (Opsional)",
        options=["Semua"] + target_levels
    )

    if selected_level != "Semua":
        heat_df = df[df["Education Level"] == selected_level]
    else:
        heat_df = df.copy()

    if heat_df.empty:
        st.warning("⚠️ Data tidak tersedia untuk pilihan ini.")
    else:
        # Membuat tabel silang
        ct = pd.crosstab(
            heat_df["Specialization"],
            heat_df["Recommended Career"]
        )

        if ct.empty:
            st.warning("⚠️ Crosstab kosong, tidak dapat ditampilkan.")
        else:
            fig, ax = plt.subplots(figsize=(14, 8))

            sns.heatmap(
                ct,
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                linewidths=0.5,
                cbar=True,
                ax=ax
            )

            ax.set_title(
                "Heatmap: Korelasi Spesialisasi vs Rekomendasi Karier",
                fontsize=16,
                fontweight="bold"
            )
            ax.set_xlabel("Rekomendasi Karier")
            ax.set_ylabel("Spesialisasi")

            st.pyplot(fig)


# -----------------------------
# TABEL RINGKASAN
# -----------------------------
st.markdown("### 📋 Tabel Persentase Keberhasilan")
st.dataframe(
    success_pct.round(2),
    use_container_width=True
)

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
