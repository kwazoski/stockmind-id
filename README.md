#  StockMind ID — Prediksi Saham Indonesia

Dashboard ML untuk prediksi harga & deteksi anomali 5 saham IDX.

## 🗂 Struktur Project

```
stockmind/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── model/                         ← dari export Colab kamu
    ├── metadata.json
    ├── BBCA_JK/
    │   ├── model_prediction.pkl
    │   ├── model_anomaly.pkl
    │   ├── scaler_X.pkl
    │   ├── scaler_y.pkl
    │   └── scaler_anomaly.pkl
    ├── BBRI_JK/ ...
    ├── TLKM_JK/ ...
    ├── GOTO_JK/ ...
    └── ASII_JK/ ...
```

##  Deploy ke Streamlit Cloud

### Step 1 — Siapkan model dari Colab
1. Download `models_saham_indonesia.zip` dari Colab
2. Extract → kamu dapat folder `model/`
3. Pastikan ada `model/metadata.json`

### Step 2 — Push ke GitHub
```bash
git init
git add .
git commit -m "initial deploy"
git remote add origin https://github.com/kwazoski/stockmind-id.git
git push -u origin main
```

>  Folder `model/` berisi file .pkl besar.  
> Kalau > 100MB, gunakan [Git LFS](https://git-lfs.github.com/):
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> ```

### Step 3 — Deploy di Streamlit Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan GitHub
3. Klik **New app**
4. Pilih repo → branch `main` → file `app.py`
5. Klik **Deploy!**

### Step 4 — Selesai! 
URL kamu: `https://USERNAME-stockmind-id.streamlit.app`

---

##  Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

##  Catatan
- Data harga realtime diambil dari Yahoo Finance
- Model .pkl **harus** diupload agar fitur prediksi ML aktif
- Tanpa model, app berjalan dalam **mode demo** (hanya tampilkan MA & harga aktual)
- Bukan saran investasi — untuk keperluan akademik/riset
