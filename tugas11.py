import streamlit as st
import numpy as np
from PIL import Image
import random

st.set_page_config(
    page_title="K-Means Color Extractor",
    page_icon="🎨",
    layout="centered"
)

st.markdown("""
<style>
.swatch-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 1rem; }
.swatch {
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
    min-width: 110px;
    font-family: comic sans;
    font-size: 13px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

st.title("🎨 K-Means Color Extractor")
st.caption("Upload an image to extract its dominant colors using K-Means clustering")

def kmeans_colors(pixels, k, max_iter=30):
    # K-Means++ initialization
    centroids = [pixels[random.randint(0, len(pixels)-1)].tolist()]
    while len(centroids) < k:
        dists = np.array([min(np.sum((p - np.array(c))**2) for c in centroids) for p in pixels])
        probs = dists / dists.sum()
        idx = np.random.choice(len(pixels), p=probs)
        centroids.append(pixels[idx].tolist())
    centroids = np.array(centroids, dtype=float)

    labels = np.zeros(len(pixels), dtype=int)
    for _ in range(max_iter):
        dists = np.array([[np.sum((p - c)**2) for c in centroids] for p in pixels])
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = pixels[mask].mean(axis=0)

    counts = np.bincount(labels, minlength=k)
    result = []
    for i in range(k):
        r, g, b = int(centroids[i][0]), int(centroids[i][1]), int(centroids[i][2])
        pct = round(counts[i] / len(pixels) * 100)
        result.append({"r": r, "g": g, "b": b, "pct": pct,"hex": f"#{r:02x}{g:02x}{b:02x}".upper()})
    return sorted(result, key=lambda x: -x["pct"])

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

k = st.select_slider("Number of colors (K)", options=[3, 4, 5, 6, 8], value=5)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("✨ Extract Palette", use_container_width=True, type="primary"):
        with st.spinner("Running K-Means clustering..."):
            img_small = img.resize((150, 150))
            pixels = np.array(img_small).reshape(-1, 3)
            sample = pixels[np.random.choice(len(pixels), min(2000, len(pixels)), replace=False)]
            colors = kmeans_colors(sample, k)

        st.subheader("Dominant Color Palette")

        # Color strip
        strip_html = '<div style="display:flex; height:24px; border-radius:8px; overflow:hidden; margin-bottom:1rem;">'
        for c in colors:
            strip_html += f'<div style="flex:{c["pct"]}; background:{c["hex"]};"></div>'
        strip_html += '</div>'
        st.markdown(strip_html, unsafe_allow_html=True)

        # Swatches
        cols = st.columns(len(colors))
        for i, c in enumerate(colors):
            brightness = 0.299*c["r"] + 0.587*c["g"] + 0.114*c["b"]
            txt = "#202424" if brightness > 128 else "#f5f5f5"
            with cols[i]:
                st.markdown(f"""
                <div style="background:{c['hex']}; border-radius:12px; padding:14px 10px;
                            text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.18);">
                  <div style="color:{txt}; font-weight:600; font-size:13px; font-family:monospace;">{c['hex']}</div>
                  <div style="color:{txt}; font-size:11px; font-family:monospace; opacity:0.85;">rgb({c['r']},{c['g']},{c['b']})</div>
                  <div style="color:{txt}; font-size:11px; opacity:0.8; margin-top:4px;">{c['pct']}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.caption("💡 Tip: Click any hex code above to copy it manually.")

st.markdown("---")
st.caption("Built with Streamlit_K-Means Clustering_Tugas 11_240034_Nena Haryadi P")