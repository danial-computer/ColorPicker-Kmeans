import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="danial 240030", page_icon="🎨", layout="centered")
st.title("danial 240030")

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_contrast_text_color(rgb):
    brightness = (rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    return "#000000" if brightness > 128 else "#FFFFFF"

# Fungsi ini di-cache agar grafik elbow tidak perlu dihitung ulang setiap kali slider digeser
@st.cache_data(show_spinner=False)
def calculate_elbow(pixels):
    inertias = []
    K_range = range(1, 11) # Menguji K dari 1 sampai 10
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init="auto", random_state=42)
        kmeans.fit(pixels)
        inertias.append(kmeans.inertia_)
    return K_range, inertias

uploaded_file = st.file_uploader(label="Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Kompresi gambar di awal agar pemrosesan lebih cepat
    img_compressed = image.copy()
    img_compressed.thumbnail((200, 200))
    img_array = np.array(img_compressed)
    pixels = img_array.reshape(-1, 3)

    st.divider()
    
    # ── METODE ELBOW ──
    st.subheader("1. Analisis Metode Elbow")
    st.write("Grafik di bawah ini menunjukkan tingkat 'distorsi' untuk setiap jumlah warna. Titik patahan (elbow) adalah rekomendasi jumlah warna yang paling optimal.")
    
    with st.spinner('Menghitung Metode Elbow (mungkin memakan waktu beberapa detik)...'):
        K_range, inertias = calculate_elbow(pixels)
        
        fig_elbow, ax_elbow = plt.subplots(figsize=(8, 3))
        ax_elbow.plot(K_range, inertias, marker='o', linestyle='-', color='teal')
        ax_elbow.set_title("Grafik Patahan (Elbow Method)", fontsize=12)
        ax_elbow.set_xlabel("Jumlah Cluster (K)", fontsize=10)
        ax_elbow.set_ylabel("Inertia (Distorsi)", fontsize=10)
        ax_elbow.set_xticks(K_range)
        ax_elbow.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig_elbow)
        plt.close(fig_elbow)

    st.divider()

    # ── SLIDER & EKSTRAKSI WARNA ──
    st.subheader("2. Ekstraksi Palet Warna")
    # Slider untuk memilih jumlah K (warna)
    num_colors = st.slider("Pilih jumlah warna yang ingin diekstrak:", min_value=2, max_value=10, value=5)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.write("**Gambar Asli**")
        st.image(image, use_container_width=True)

    with col_right:
        st.write(f"**Top {num_colors} Warna Dominan**")
        
        # K-Means final menggunakan jumlah warna dari slider
        kmeans = KMeans(n_clusters=num_colors, init='k-means++', n_init="auto", random_state=42)
        labels = kmeans.fit_predict(pixels)
        centroids = kmeans.cluster_centers_.astype(int)

        counts = np.bincount(labels)
        percentages = counts / len(labels)
        sorted_indices = np.argsort(percentages)[::-1]
        sorted_colors = centroids[sorted_indices]
        sorted_percentages = percentages[sorted_indices]

        for i, (color, percent) in enumerate(zip(sorted_colors, sorted_percentages)):
            hex_code = rgb_to_hex(color)
            text_color = get_contrast_text_color(color)
            pct_string = f"{percent * 100:.1f}%"
            st.markdown(
                f"""
                <div style="
                    background-color: {hex_code}; 
                    color: {text_color};
                    padding: 10px 15px; 
                    border-radius: 8px; 
                    margin-bottom: 8px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-weight: bold;
                    box-shadow: 1px 1px 4px rgba(0,0,0,0.15);
                ">
                    <span>{hex_code}</span>
                    <span style="opacity: 0.85; font-size: 0.9em;">{pct_string}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # ── SCATTER PLOT ──
    st.subheader("3. Visualisasi Proses K-Means (PCA 2D)")
    
    N_SAMPLE = 800
    sample_idx = np.random.choice(len(pixels), size=min(N_SAMPLE, len(pixels)), replace=False)
    sample_pixels = pixels[sample_idx]
    
    pca = PCA(n_components=2)
    pixels_2d = pca.fit_transform(sample_pixels)
    centroids_2d = pca.transform(centroids)

    colors_normalized = sample_pixels / 255.0
    centroid_colors = np.clip(centroids / 255.0, 0, 1)

    fig_pca, ax_pca = plt.subplots(figsize=(8, 4))
    ax_pca.scatter(pixels_2d[:, 0], pixels_2d[:, 1], c=colors_normalized, s=6, alpha=0.55, linewidths=0)
    ax_pca.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=centroid_colors, s=250, marker='X', edgecolors='black', linewidths=1.2, zorder=5, label="Centroid")

    ax_pca.set_title(f"Distribusi Piksel ({num_colors} Cluster)", fontsize=12)
    ax_pca.set_xlabel("PCA Komponen 1", fontsize=10)
    ax_pca.set_ylabel("PCA Komponen 2", fontsize=10)
    ax_pca.legend(fontsize=9)
    plt.tight_layout()

    st.pyplot(fig_pca)
    plt.close(fig_pca)