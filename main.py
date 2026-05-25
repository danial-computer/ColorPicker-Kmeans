import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Konfigurasi halaman dibuat menjadi WIDE
st.set_page_config(page_title="danial 240030", page_icon="🎨", layout="wide")

# CSS tambahan untuk sedikit mengurangi padding atas bawaan Streamlit
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

@st.cache_data(show_spinner=False)
def calculate_elbow(pixels):
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init="auto", random_state=42)
        kmeans.fit(pixels)
        inertias.append(kmeans.inertia_)
    return K_range, inertias

# ── 1. SIDEBAR (PENGATURAN) ──
with st.sidebar:
    st.title("⚙️ Pengaturan")
    num_colors = st.slider("Jumlah Warna (K)", min_value=2, max_value=10, value=5, help="Geser untuk menentukan berapa banyak warna yang ingin diekstrak.")
    st.divider()
    st.caption("Aplikasi: danial 240030")

# ── 2. MAIN HEADER ──
st.title("🎨 AI Color Palette Extractor")
st.markdown("Ekstrak warna dominan dari gambarmu menggunakan algoritma **K-Means Clustering**.")

uploaded_file = st.file_uploader("Unggah gambar (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Kompresi piksel untuk efisiensi komputasi
    img_compressed = image.copy()
    img_compressed.thumbnail((200, 200))
    img_array = np.array(img_compressed)
    pixels = img_array.reshape(-1, 3)
    
    # ── 3. TAMPILAN GAMBAR & PALET ──
    st.write("---")
    col_img, col_palette = st.columns([1, 2], gap="large")
    
    with col_img:
        st.image(image, caption="Gambar Original", use_container_width=True)
        
    with col_palette:
        st.subheader(f"✨ Palet {num_colors} Warna Dominan")
        
        with st.spinner('Mengekstrak palet warna...'):
            kmeans = KMeans(n_clusters=num_colors, init='k-means++', n_init="auto", random_state=42)
            labels = kmeans.fit_predict(pixels)
            centroids = kmeans.cluster_centers_.astype(int)

            counts = np.bincount(labels)
            percentages = counts / len(labels)
            sorted_indices = np.argsort(percentages)[::-1]
            sorted_colors = centroids[sorted_indices]
            sorted_percentages = percentages[sorted_indices]

            # Membuat kolom horizontal sesuai jumlah warna yang dipilih
            color_cols = st.columns(num_colors)
            for col, color, percent in zip(color_cols, sorted_colors, sorted_percentages):
                hex_code = rgb_to_hex(color)
                pct_string = f"{percent * 100:.1f}%"
                
                with col:
                    # Desain Color Card UI
                    st.markdown(f"""
                        <div style="
                            background-color: {hex_code};
                            height: 120px;
                            border-radius: 12px;
                            margin-bottom: 10px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
                            transition: transform 0.2s;
                        ">
                        </div>
                        <div style="text-align: center; font-family: monospace;">
                            <p style="margin: 0; font-weight: 800; font-size: 1.1em; color: #333;">{hex_code}</p>
                            <p style="margin: 0; font-size: 0.9em; color: #666;">{pct_string}</p>
                        </div>
                    """, unsafe_allow_html=True)

    st.write("---")

    # ── 4. ANALISIS TEKNIS (TABS) ──
    st.subheader("🔬 Analisis Teknis (Di Balik Layar)")
    tab_elbow, tab_pca = st.tabs(["📉 Metode Elbow", "🌌 Scatter Plot (PCA 2D)"])
    
    with tab_elbow:
        st.markdown("Grafik ini menunjukkan evaluasi nilai *inertia* (seberapa rapat cluster). Titik patahan (elbow) membantu menentukan jumlah $K$ yang optimal.")
        with st.spinner('Menghitung nilai Elbow...'):
            K_range, inertias = calculate_elbow(pixels)
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4))
            ax_elbow.plot(K_range, inertias, marker='o', linestyle='-', color='#FF4B4B', linewidth=2)
            ax_elbow.set_title("Evaluasi K-Means dengan Elbow Method")
            ax_elbow.set_xlabel("Jumlah Cluster (K)")
            ax_elbow.set_ylabel("Inertia")
            ax_elbow.set_xticks(K_range)
            ax_elbow.grid(True, linestyle='--', alpha=0.5)
            # Menghilangkan border atas dan kanan agar lebih estetik
            ax_elbow.spines['top'].set_visible(False)
            ax_elbow.spines['right'].set_visible(False)
            
            st.pyplot(fig_elbow)
            plt.close(fig_elbow)

    with tab_pca:
        st.markdown(f"Visualisasi distribusi piksel dan *centroid* pada dimensi ruang yang direduksi menjadi 2D menggunakan PCA untuk **$K={num_colors}$**.")
        
        N_SAMPLE = 1000
        sample_idx = np.random.choice(len(pixels), size=min(N_SAMPLE, len(pixels)), replace=False)
        sample_pixels = pixels[sample_idx]
        
        pca = PCA(n_components=2)
        pixels_2d = pca.fit_transform(sample_pixels)
        centroids_2d = pca.transform(centroids)

        colors_normalized = sample_pixels / 255.0
        centroid_colors = np.clip(centroids / 255.0, 0, 1)

        fig_pca, ax_pca = plt.subplots(figsize=(10, 4))
        ax_pca.scatter(pixels_2d[:, 0], pixels_2d[:, 1], c=colors_normalized, s=15, alpha=0.6, linewidths=0)
        ax_pca.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=centroid_colors, s=300, marker='X', edgecolors='white', linewidths=2, zorder=5, label="Centroid")

        ax_pca.set_title(f"Distribusi Piksel di Ruang PCA", fontsize=12)
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")
        ax_pca.spines['top'].set_visible(False)
        ax_pca.spines['right'].set_visible(False)
        ax_pca.legend()
        
        st.pyplot(fig_pca)
        plt.close(fig_pca)