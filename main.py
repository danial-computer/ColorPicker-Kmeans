import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Judul halaman dan aplikasi diubah menjadi "danial 240030"
st.set_page_config(page_title="danial 240030", page_icon="🎨", layout="centered")
st.title("danial 240030")

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_contrast_text_color(rgb):
    brightness = (rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    return "#000000" if brightness > 128 else "#FFFFFF"

uploaded_file = st.file_uploader(label="Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ── BARIS 1: Gambar | Analisis K-Means ──
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Gambar")
        st.image(image, use_container_width=True)

    with col_right:
        st.subheader("Analisis K-Means")
        status_placeholder = st.empty()
        status_placeholder.info("⚡ Mengompresi dan memproses piksel...")

        # Kompresi
        img_compressed = image.copy()
        img_compressed.thumbnail((200, 200))
        img_array = np.array(img_compressed)
        pixels = img_array.reshape(-1, 3)

        # K-Means final pada semua piksel
        N_CLUSTERS = 5
        kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', n_init="auto", random_state=42)
        labels = kmeans.fit_predict(pixels)
        centroids = kmeans.cluster_centers_.astype(int)

        counts = np.bincount(labels)
        percentages = counts / len(labels)
        sorted_indices = np.argsort(percentages)[::-1]
        sorted_colors = centroids[sorted_indices]
        sorted_percentages = percentages[sorted_indices]

        status_placeholder.empty()
        st.success("Berikut 5 major colornya:")

        for i, (color, percent) in enumerate(zip(sorted_colors, sorted_percentages)):
            hex_code = rgb_to_hex(color)
            text_color = get_contrast_text_color(color)
            pct_string = f"{percent * 100:.1f}%"
            st.markdown(
                f"""
                <div style="
                    background-color: {hex_code}; 
                    color: {text_color};
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-bottom: 8px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-weight: bold;
                    box-shadow: 1px 1px 4px rgba(0,0,0,0.15);
                ">
                    <span>Warna {i+1} : {hex_code}</span>
                    <span style="opacity: 0.85; font-size: 0.9em;">Proporsi: {pct_string}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ── BARIS 2: Scatter Plot K-Means (full width, di bawah kedua kolom) ──
    st.divider()
    st.subheader("Visualisasi Proses K-Means Clustering")

    # Sample piksel untuk visualisasi
    N_SAMPLE = 800
    sample_idx = np.random.choice(len(pixels), size=min(N_SAMPLE, len(pixels)), replace=False)
    sample_pixels = pixels[sample_idx]
    sample_labels = kmeans.predict(sample_pixels)

    # PCA: 3D RGB → 2D untuk scatter
    pca = PCA(n_components=2)
    pixels_2d = pca.fit_transform(sample_pixels)
    centroids_2d = pca.transform(centroids)

    # Warna asli tiap piksel
    colors_normalized = sample_pixels / 255.0
    centroid_colors = np.clip(centroids / 255.0, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Scatter titik piksel dengan warna aslinya
    ax.scatter(
        pixels_2d[:, 0], pixels_2d[:, 1],
        c=colors_normalized,
        s=6, alpha=0.55, linewidths=0
    )

    # Centroid dengan marker 'X' besar + border hitam
    ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c=centroid_colors,
        s=250, marker='X',
        edgecolors='black', linewidths=1.2,
        zorder=5, label="Centroid"
    )

    ax.set_title("Distribusi Piksel dalam Ruang PCA (Hasil Akhir Clustering)", fontsize=12)
    ax.set_xlabel("PCA Komponen 1", fontsize=10)
    ax.set_ylabel("PCA Komponen 2", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Tanda X adalah centroid akhir dari tiap cluster warna"
    )