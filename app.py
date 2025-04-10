import streamlit as st
import numpy as np
import h5py
import tensorflow as tf
import requests
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
from huggingface_hub import hf_hub_download
import requests
import os
import streamlit as st
from huggingface_hub import login
from dotenv import load_dotenv
import streamlit as st
st.set_page_config(page_title="HSI Cancer Detector", layout="centered")
st.title("🔬 Skin Cancer Detection using Hyperspectral Imaging")

REPO_ID = "Nishtha001/CNNAE"
FILENAME = "cnn_model.keras"

# ✅ Get token from Streamlit secrets
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("❌ Hugging Face token not found in secrets!")
    st.stop()

# ✅ Force download from Hub (not cache)
try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=hf_token,
        local_dir="/tmp/hf_model",        # Force download to tmp
        local_dir_use_symlinks=False,     # Avoid symlinks for Streamlit
        force_download=True               # 🚨 Force fresh download
    )
    st.success("✅ Model downloaded from Hugging Face.")
except Exception as e:
    st.error(f"❌ Failed to download model: {e}")
    st.stop()

# ✅ Load model
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

TARGET_SIZE = (128, 128)
NUM_BANDS = 31
THRESHOLD = 0.2



# Anomaly score = MSE + (1 - SSIM)
def compute_anomaly_score(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    ssim_score = ssim(original, reconstructed, data_range=original.max() - original.min(), channel_axis=2)
    return mse + (1 - ssim_score)

# Streamlit UI

uploaded_file = st.file_uploader("📤 Upload a Hyperspectral `.mat` file (MATLAB v7.3)", type=["mat"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    try:
        with h5py.File(uploaded_file, 'r') as file:
            keys = list(file.keys())
            st.write("📁 Keys found in file:", keys)

            # Try to pick the most likely image key
            possible_keys = [k for k in keys if 'hsi' in k.lower() or 'image' in k.lower()]
            image_key = possible_keys[0] if possible_keys else keys[0]
            st.write(f"🔍 Using data from key: `{image_key}`")

            hsi_image = np.array(file[image_key])

            # Transpose if needed (v7.3 stores differently)
            if hsi_image.shape[0] <= NUM_BANDS:
                hsi_image = np.transpose(hsi_image, (1, 2, 0))

        # Normalize
        hsi_image = hsi_image.astype(np.float32)
        hsi_image = (hsi_image - np.min(hsi_image)) / (np.max(hsi_image) - np.min(hsi_image))

        # Resize if necessary
        if hsi_image.shape[:2] != TARGET_SIZE:
            st.warning(f"⚠️ Resizing image from {hsi_image.shape[:2]} to {TARGET_SIZE}.")
            resized_image = np.zeros((*TARGET_SIZE, NUM_BANDS), dtype=np.float32)
            for i in range(NUM_BANDS):
                resized_image[:, :, i] = resize(hsi_image[:, :, i], TARGET_SIZE, anti_aliasing=True)
            hsi_image = resized_image

        # Add batch dimension
        input_image = np.expand_dims(hsi_image, axis=0)

        # Predict
        reconstructed = model.predict(input_image)
        reconstructed_image = np.squeeze(reconstructed)

        # Anomaly score
        score = compute_anomaly_score(hsi_image, reconstructed_image)

        if score > THRESHOLD:
            st.error(f"⚠️ Likely **Cancerous**.\n\n🧪 Anomaly Score: `{score:.4f}`")
        else:
            st.success(f"✅ Likely **Healthy**.\n\n🧪 Anomaly Score: `{score:.4f}`")

        # Visualization
        st.subheader("📊 Original vs Reconstructed (Band 15)")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        band_idx = min(15, hsi_image.shape[2] - 1)

        axes[0].imshow(hsi_image[:, :, band_idx], cmap='viridis')
        axes[0].set_title("Original Band 15")
        axes[1].imshow(reconstructed_image[:, :, band_idx], cmap='viridis')
        axes[1].set_title("Reconstructed Band 15")
        for ax in axes: ax.axis("off")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing the file:\n\n`{str(e)}`")
