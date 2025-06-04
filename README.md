# 🧬 Skin Cancer Detection using Hyperspectral Imaging

Live demo: [https://skincancerusingautoencoder001.streamlit.app/](https://skincancerusingautoencoder001.streamlit.app/)

RGB to HSI conversion :[https://github.com/Nishtha077/rgb_to_hsi](https://github.com/Nishtha077/rgb_to_hsi)


This Streamlit app detects potential cancerous regions in hyperspectral skin images using a **CNN Autoencoder**. It works on the principle that abnormal (cancerous) areas have a **higher reconstruction error** when compared to healthy tissue.

---

## 🧠 Model Details

- **Input:** Hyperspectral image (256 × 256 × 31)
- **Model:** CNN Autoencoder
- **Output:** Reconstructed image + Error map
- **Detection Rule:**  
  - Reconstruction error > **0.2** → Marked as **cancerous**

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/skin_cancer_hsi.git
cd skin_cancer_hsi
pip install -r requirements.txt
streamlit run app.py
