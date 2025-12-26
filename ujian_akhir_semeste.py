import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# --- CONFIGURASI HALAMAN (Wajib di paling atas) ---
st.set_page_config(page_title="Analisis Pendidikan UAS", layout="wide")

# 1. PENGATURAN PATH FILE
file_path = 'career_dataset_large.xlsx'

# 2. FUNGSI LOAD DATA DENGAN CACHE
@st.cache_data
def load_data():
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error: Tidak dapat menemukan file '{file_path}'.")
        return None

# 3. MEMANGGIL DATA KE DALAM VARIABEL
df = load_data()

# 4. CEK JIKA DATA BERHASIL DI-LOAD
if df is not None:
    st.success("Data berhasil dimuat!")

    # 2. PEMBERSIHAN DATA
    df['Education Level'] = df['Education Level'].str.strip()

    # 3. FILTER & MENGURUTKAN
    target_levels = ["Intermediate", "Master's", "Bachelor's", "Matric", "PhD"]
    df_filtered = df[df['Education Level'].isin(target_levels)].copy()

    # 4. MEMBUAT TABEL RINGKASAN
    summary_table = df_filtered['Education Level'].value_counts().reindex(target_levels).reset_index()
    summary_table.columns = ['Education Level', 'Total Data']

    # 5. MENAMPILKAN HASIL (Ganti print & display)
    st.write("### Tabel Jenjang Pendidikan (Sesuai Filter):")
    st.dataframe(summary_table)

    st.write("### Contoh Data Mentah Setelah Filter:")
    st.dataframe(df_filtered.head(20))

    # Kolom yang ingin dianalisis
    cols = ['Specialization', 'Skills', 'Certifications', 'CGPA/Percentage', 'Recommended Career']
    results = []

    for level in target_levels:
        level_df = df[df['Education Level'] == level]
        for col in cols:
            counts = level_df[col].value_counts()
            if not counts.empty:
                results.append({
                    'Jenjang': level,
                    'Kolom': col,
                    'Jumlah Unik (Beda)': level_df[col].nunique(),
                    'Nilai Terpopuler': counts.index[0],
                    'Frekuensi (Sama)': counts.iloc[0]
                })

    # Menampilkan hasil dalam bentuk tabel
    summary_df = pd.DataFrame(results)
    st.write("### Summary Statistics:")
    st.dataframe(summary_df)

    # 2. PEMBERSIHAN & PEMBUATAN KOLOM STATUS
    df['Status_Keberhasilan'] = np.where(df['CGPA/Percentage'] >= 80, 'Berhasil', 'Gagal')

    # 3. FUNGSI UNTUK MENGAMBIL SEMUA SKILL
    def get_all_skills(subset):
        all_s = []
        for s in subset.dropna():
            all_s.extend([i.strip() for i in str(s).split(',')])
        return pd.Series(all_s).value_counts()

    # 4. MENGHITUNG SKILL PER KELOMPOK
    skill_berhasil = get_all_skills(df[df['Status_Keberhasilan'] == 'Berhasil']['Skills'])
    skill_gagal = get_all_skills(df[df['Status_Keberhasilan'] == 'Gagal']['Skills'])

    # 5. MENGHITUNG SELISIH
    diff_skill = (skill_berhasil - skill_gagal).sort_values(ascending=False).head(10)

    # 6. VISUALISASI SKILL PEMBEDA
    fig_skill, ax_skill = plt.subplots(figsize=(12, 7))
    diff_skill.plot(kind='barh', color='seagreen', ax=ax_skill)
    ax_skill.set_title('Top 10 Skill Pembeda', fontsize=14, fontweight='bold')
    ax_skill.invert_yaxis()
    st.pyplot(fig_skill)

    # VISUALISASI BOXPLOT
    fig_box, ax_box = plt.subplots(figsize=(15, 7))
    sns.boxplot(data=df, x='Recommended Career', y='CGPA/Percentage', palette='Set3', ax=ax_box)
    plt.xticks(rotation=45)
    st.pyplot(fig_box)

    # HEATMAP
    fig_heat, ax_heat = plt.subplots(figsize=(14, 8))
    ct = pd.crosstab(df['Specialization'], df['Recommended Career'])
    sns.heatmap(ct, annot=True, cmap='YlGnBu', fmt='d', cbar=True, ax=ax_heat)
    st.pyplot(fig_heat)

    # STACKED BAR CHART KARIER
    fig_career, ax_career = plt.subplots(figsize=(15, 8))
    career_dist = pd.crosstab(df['Education Level'], df['Recommended Career'], normalize='index') * 100
    career_dist.plot(kind='bar', stacked=True, colormap='tab20', ax=ax_career)
    plt.legend(title='Karier', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_career)

    # 3. FUNGSI UNTUK MENGHITUNG TOP ITEMS
    def get_top_items(dataframe, edu_level, status, column, top_n=5):
        subset = dataframe[(dataframe['Education Level'] == edu_level) &
                           (dataframe['Status_Keberhasilan'] == status)][column].dropna()
        all_items = []
        for entry in subset:
            items = [item.strip() for item in str(entry).split(',')]
            all_items.extend(items)
        counts = Counter(all_items).most_common(top_n)
        return pd.DataFrame(counts, columns=['Item', 'Count'])

    # 4. VISUALISASI UNTUK 5 JENJANG PENDIDIKAN (Looping)
    for jenjang in list_jenjang:
        if jenjang in df['Education Level'].unique():
            fig_loop, axes = plt.subplots(4, 2, figsize=(16, 24))
            fig_loop.suptitle(f'Analisis Profil Lengkap: {jenjang}', fontsize=22, fontweight='bold', y=1.01)

            # Baris 1: Specialization
            d_sp_b = get_top_items(df, jenjang, 'Berhasil', 'Specialization')
            if not d_sp_b.empty: sns.barplot(data=d_sp_b, x='Count', y='Item', ax=axes[0, 0], palette='Greens_r')
            d_sp_g = get_top_items(df, jenjang, 'Gagal', 'Specialization')
            if not d_sp_g.empty: sns.barplot(data=d_sp_g, x='Count', y='Item', ax=axes[0, 1], palette='Reds_r')

            # Baris 2: Skills
            d_s_b = get_top_items(df, jenjang, 'Berhasil', 'Skills')
            if not d_s_b.empty: sns.barplot(data=d_s_b, x='Count', y='Item', ax=axes[1, 0], palette='Greens_r')
            d_s_g = get_top_items(df, jenjang, 'Gagal', 'Skills')
            if not d_s_g.empty: sns.barplot(data=d_s_g, x='Count', y='Item', ax=axes[1, 1], palette='Reds_r')

            # Baris 3: Certifications
            d_c_b = get_top_items(df, jenjang, 'Berhasil', 'Certifications')
            if not d_c_b.empty: sns.barplot(data=d_c_b, x='Count', y='Item', ax=axes[2, 0], palette='Greens_r')
            d_c_g = get_top_items(df, jenjang, 'Gagal', 'Certifications')
            if not d_c_g.empty: sns.barplot(data=d_c_g, x='Count', y='Item', ax=axes[2, 1], palette='Reds_r')

            # Baris 4: CGPA
            sub_b = df[(df['Education Level'] == jenjang) & (df['Status_Keberhasilan'] == 'Berhasil')]
            sns.histplot(sub_b['CGPA/Percentage'], ax=axes[3, 0], color='green', kde=True)
            sub_g = df[(df['Education Level'] == jenjang) & (df['Status_Keberhasilan'] == 'Gagal')]
            sns.histplot(sub_g['CGPA/Percentage'], ax=axes[3, 1], color='red', kde=True)

            plt.tight_layout()
            st.pyplot(fig_loop)

    # 5. VISUALISASI PIE & STACKED BAR AKHIR
    dist_counts = df['Education Level'].value_counts(normalize=True) * 100
    dist_df = dist_counts.reindex(target_levels).reset_index()
    dist_df.columns = ['Education Level', 'Percentage']

    success_counts_final = df.groupby(['Education Level', 'Status_Keberhasilan'], observed=True).size().unstack(fill_value=0)
    success_pct_final = success_counts_final.div(success_counts_final.sum(axis=1), axis=0) * 100

    fig_final, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.pie(dist_df['Percentage'], labels=dist_df['Education Level'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    success_pct_final.reindex(target_levels).plot(kind='bar', stacked=True, ax=ax2, color=['#2ca02c', '#d62728'])
    plt.tight_layout()
    st.pyplot(fig_final)

    # WORDCLOUD AKHIR
    st.write("### Word Cloud Skills:")
    all_skills_text = " ".join(df['Skills'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_skills_text)
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
