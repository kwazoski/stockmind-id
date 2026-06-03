import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockMind ID — Prediksi Saham Indonesia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"], [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main, .block-container {
    font-family: 'Segoe UI', Tahoma, Geneva, sans-serif !important;
    font-size: 14px !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
  }
  [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: #f5f5f5 !important;
    border-right: 1px solid #e0e0e0 !important;
  }
  [data-testid="stSidebar"] * { color: #1a1a1a !important; }
  .stSelectbox > div > div { background: #fff !important; color: #1a1a1a !important; border: 1px solid #ccc !important; }
  .stDateInput input { background: #fff !important; color: #1a1a1a !important; border: 1px solid #ccc !important; }
  [data-testid="stTabs"] button { color: #555 !important; }
  [data-testid="stTabs"] button[aria-selected="true"] { color: #1a1a1a !important; border-bottom: 2px solid #1a1a1a !important; }
  hr { border-color: #e0e0e0 !important; }
  .stCaption, [data-testid="stCaptionContainer"] { color: #777 !important; }
  [data-testid="stDataFrame"] * { color: #1a1a1a !important; }
  footer, #MainMenu { visibility: hidden; }

  .page-title { font-size: 2.2rem; font-weight: 800; color: #0a0a0a; letter-spacing: -0.02em; margin-bottom: 4px; }
  .page-sub   { font-size: 0.88rem; color: #777; margin-bottom: 1.8rem; }

  .section-title {
    font-size: 1.2rem; font-weight: 700; color: #111;
    margin: 28px 0 6px 0; border-left: 4px solid #111;
    padding-left: 10px;
  }
  .section-label {
    font-size: 0.75rem; font-weight: 700; color: #555;
    text-transform: uppercase; letter-spacing: 0.12em;
    border-bottom: 1px solid #e0e0e0; padding-bottom: 7px;
    margin: 24px 0 14px 0;
  }
  .kpi-box {
    border: 1px solid #e0e0e0; border-radius: 6px;
    padding: 16px 20px; background: #fafafa; margin-bottom: 4px;
  }
  .kpi-label { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; font-weight: 500; }
  .kpi-value { font-size: 1.5rem; font-weight: 700; color: #0a0a0a; }
  .kpi-value.up   { color: #1a7a3c; }
  .kpi-value.down { color: #c0392b; }
  .kpi-value.muted { color: #222; font-size: 1.05rem; font-weight: 600; }

  .info-box {
    background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 18px 22px; margin-bottom: 14px;
  }
  .info-box h4 { font-size: 0.85rem; font-weight: 700; color: #333; margin: 0 0 6px 0; }
  .info-box p  { font-size: 0.82rem; color: #555; margin: 0; line-height: 1.6; }

  .member-card {
    border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 20px; background: #fafafa; text-align: center;
  }
  .member-name { font-size: 1rem; font-weight: 700; color: #111; margin-bottom: 4px; }
  .member-role { font-size: 0.78rem; color: #888; }

  .method-card {
    border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 16px 18px; background: #fafafa; margin-bottom: 10px;
  }
  .method-title { font-size: 0.9rem; font-weight: 700; color: #111; margin-bottom: 4px; }
  .method-desc  { font-size: 0.8rem; color: #555; line-height: 1.5; }

  .nav-item { font-size: 0.82rem; font-weight: 600; color: #333; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TICKERS = {
    'BBCA.JK': 'Bank BCA',
    'BBRI.JK': 'Bank BRI',
    'TLKM.JK': 'Telkom Indonesia',
    'GOTO.JK': 'GoTo Group',
    'ASII.JK': 'Astra International',
}
FEATURES = ['Open','High','Low','Close','Volume','MA7','MA30',
            'Daily_Return','Volatility','Volume_Change','High_Low_Gap','Open_Close_Gap']
ANOMALY_FEATURES = ['Close','Volume','Daily_Return','Volatility','High_Low_Gap']
MODEL_DIR = "model"

TEAM_MEMBERS = [
    {"name": "Anggota 1", "role": "Data Engineer"},
    {"name": "Anggota 2", "role": "ML Engineer"},
    {"name": "Anggota 3", "role": "Data Analyst"},
]

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.reset_index(inplace=True)
    df['MA7']            = df['Close'].rolling(7).mean()
    df['MA30']           = df['Close'].rolling(30).mean()
    df['Daily_Return']   = df['Close'].pct_change()
    df['Volatility']     = df['Daily_Return'].rolling(7).std()
    df['Volume_Change']  = df['Volume'].pct_change()
    df['High_Low_Gap']   = df['High'] - df['Low']
    df['Open_Close_Gap'] = df['Close'] - df['Open']
    df['Target']         = df['Close'].shift(-1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def model_ok(ticker):
    tc = ticker.replace('.', '_')
    folder = os.path.join(MODEL_DIR, tc)
    files = ['model_prediction.pkl','scaler_X.pkl','scaler_y.pkl',
             'model_anomaly.pkl','scaler_anomaly.pkl']
    return all(os.path.exists(os.path.join(folder, f)) for f in files)

def load_models(ticker):
    tc = ticker.replace('.', '_')
    folder = os.path.join(MODEL_DIR, tc)
    return (
        joblib.load(f'{folder}/model_prediction.pkl'),
        joblib.load(f'{folder}/scaler_X.pkl'),
        joblib.load(f'{folder}/scaler_y.pkl'),
        joblib.load(f'{folder}/model_anomaly.pkl'),
        joblib.load(f'{folder}/scaler_anomaly.pkl'),
    )

def load_meta():
    p = os.path.join(MODEL_DIR, 'metadata.json')
    return json.load(open(p)) if os.path.exists(p) else {}

def predict(df, model, sc_X, sc_y):
    X_sc = sc_X.transform(df[FEATURES].values)
    return sc_y.inverse_transform(model.predict(X_sc).reshape(-1,1)).ravel()

def detect_anomaly(df, iso, sc_an):
    X_sc = sc_an.transform(df[ANOMALY_FEATURES].copy())
    z    = np.abs(stats.zscore(X_sc))
    zf   = (z > 3).any(axis=1).astype(int)
    isof = (iso.predict(X_sc) == -1).astype(int)
    return zf, isof, ((zf + isof) >= 2).astype(int)

def plain_chart(**kwargs):
    d = dict(
        template='plotly_white', paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        height=400, margin=dict(l=10,r=10,t=30,b=10),
        font=dict(family='Segoe UI', size=11, color='#444'),
        legend=dict(orientation='h', y=1.05, x=0),
        xaxis=dict(gridcolor='#f0f0f0', linecolor='#ddd'),
        yaxis=dict(gridcolor='#f0f0f0', linecolor='#ddd'),
    )
    d.update(kwargs)
    return d

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**StockMind ID**")
    st.caption("Prediksi Saham Indonesia")
    st.divider()

    page = st.radio(
        "Navigasi",
        [" Home", " Dataset Overview", " Prediction & Analysis", " Visualization", "ℹ About"],
        label_visibility="collapsed"
    )
    st.divider()

    if page in [" Dataset Overview", " Prediction & Analysis", " Visualization"]:
        st.markdown("**Pengaturan**")
        ticker = st.selectbox("Saham", list(TICKERS.keys()),
                              format_func=lambda x: f"{TICKERS[x]} ({x})")
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Dari", value=datetime(2020,1,1), label_visibility="collapsed")
        with col_b:
            end_date = st.date_input("Sampai", value=datetime.today(), label_visibility="collapsed")

        if page == "🔮 Prediction & Analysis":
            forecast_days = st.slider("Forecast (hari)", 1, 30, 7)

        has_model = model_ok(ticker)
        st.divider()
        if has_model:
            st.success(" Model tersedia")
        else:
            st.warning(" Model belum diupload")
            st.caption("Letakkan folder `model/` di root project.")
    else:
        ticker = 'BBCA.JK'
        start_date = datetime(2020,1,1)
        end_date = datetime.today()
        forecast_days = 7
        has_model = model_ok(ticker)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == " Home":
    st.markdown('<div class="page-title"> StockMind Indonesia</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Sistem Prediksi Harga & Deteksi Anomali Saham IDX berbasis Machine Learning</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Tentang Proyek</div>', unsafe_allow_html=True)
    st.markdown("""
    Proyek ini merupakan implementasi **data mining dan machine learning** untuk menganalisis
    pergerakan harga saham di Bursa Efek Indonesia (BEI/IDX). Sistem ini mampu melakukan
    prediksi harga penutupan hari berikutnya serta mendeteksi anomali harga secara otomatis
    menggunakan beberapa metode statistik dan ML.
    """)

    st.markdown('<div class="section-label">Fitur Utama</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    fitur = [
        ("", "Prediksi Harga", "Prediksi harga penutupan menggunakan Linear Regression, Random Forest, XGBoost, dan SVR."),
        ("", "Deteksi Anomali", "Identifikasi pergerakan harga tidak wajar dengan Z-Score, Isolation Forest, dan LOF."),
        ("", "Visualisasi", "Grafik interaktif prediksi vs aktual, anomali, dan perbandingan performa model."),
        ("", "5 Saham IDX", "Mencakup BBCA, BBRI, TLKM, GOTO, dan ASII dengan data dari 2020 hingga sekarang."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], fitur):
        with col:
            st.markdown(f"""<div class="info-box">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Saham yang Dianalisis</div>', unsafe_allow_html=True)
    meta = load_meta()
    tmeta = meta.get('tickers', {})
    rows = []
    for t, name in TICKERS.items():
        info = tmeta.get(t, {})
        rows.append({
            'Ticker': t,
            'Perusahaan': name,
            'Best Model': info.get('best_pred_name', '—'),
            'R²': f"{info.get('r2', 0):.4f}" if info else '—',
            'MAPE': f"{info.get('mape', 0):.2f}%" if info else '—',
            'Anomali Terdeteksi': info.get('n_anomaly', '—'),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">Identitas Anggota</div>', unsafe_allow_html=True)
    cols = st.columns(len(TEAM_MEMBERS))
    for col, m in zip(cols, TEAM_MEMBERS):
        with col:
            st.markdown(f"""<div class="member-card">
                <div style="font-size:2rem">👤</div>
                <div class="member-name">{m['name']}</div>
                <div class="member-role">{m['role']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Periode Data</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    for col, lbl, val in zip([p1,p2,p3],
        ['Mulai Data','Akhir Data','Sumber Data'],
        ['1 Januari 2020','2 Juni 2026','Yahoo Finance (yfinance)']):
        with col:
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value muted">{val}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Dataset Overview":
    st.markdown('<div class="page-title"> Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{TICKERS[ticker]} · {ticker}</div>', unsafe_allow_html=True)

    with st.spinner("Mengambil data..."):
        df = fetch_data(ticker, str(start_date), str(end_date))

    if df.empty:
        st.error("Gagal mengambil data.")
        st.stop()

    # Info dataset
    st.markdown('<div class="section-label">Informasi Dataset</div>', unsafe_allow_html=True)
    i1, i2, i3, i4 = st.columns(4)
    latest = df.iloc[-1]
    for col, lbl, val in zip([i1,i2,i3,i4],
        ['Jumlah Data (Baris)','Jumlah Fitur','Periode','Harga Terakhir'],
        [f"{len(df):,}", str(len(FEATURES)), f"{df['Date'].iloc[0].strftime('%d/%m/%Y')} — {df['Date'].iloc[-1].strftime('%d/%m/%Y')}", f"Rp {latest['Close']:,.0f}"]):
        with col:
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value muted">{val}</div>
            </div>""", unsafe_allow_html=True)

    # Statistik sederhana
    st.markdown('<div class="section-label">Statistik Sederhana</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, lbl, val in zip([s1,s2,s3,s4],
        ['Harga Tertinggi','Harga Terendah','Rata-rata Close','Std Deviasi'],
        [f"Rp {df['High'].max():,.0f}", f"Rp {df['Low'].min():,.0f}",
         f"Rp {df['Close'].mean():,.0f}", f"Rp {df['Close'].std():,.0f}"]):
        with col:
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value muted">{val}</div>
            </div>""", unsafe_allow_html=True)

    # Visualisasi data
    st.markdown('<div class="section-label">Visualisasi Data</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='#1a1a1a', width=1.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'],   name='MA7',   line=dict(color='#e67e22', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA30'],  name='MA30',  line=dict(color='#2980b9', width=1, dash='dash')))
    fig.update_layout(**plain_chart(yaxis_title='Harga (Rp)', title=f'{TICKERS[ticker]} — Harga Close + Moving Average'))
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='#bdc3c7'))
    fig_vol.update_layout(**plain_chart(height=250, yaxis_title='Volume', title='Volume Transaksi'))
    st.plotly_chart(fig_vol, use_container_width=True)

    # Tabel data mentah
    st.markdown('<div class="section-label">Data Mentah (30 Hari Terakhir)</div>', unsafe_allow_html=True)
    disp = df[['Date','Open','High','Low','Close','Volume','Daily_Return']].tail(30).copy()
    disp['Date'] = pd.to_datetime(disp['Date']).dt.strftime('%d %b %Y')
    disp['Daily_Return'] = disp['Daily_Return'].map(lambda x: f"{x*100:.2f}%")
    for c in ['Open','High','Low','Close']:
        disp[c] = disp[c].map(lambda x: f"Rp {x:,.0f}")
    disp['Volume'] = disp['Volume'].map(lambda x: f"{x/1e6:.1f}M")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">Statistik Deskriptif Lengkap</div>', unsafe_allow_html=True)
    st.dataframe(df[['Close','Volume','Daily_Return','Volatility','High_Low_Gap']].describe().round(4),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction & Analysis":
    st.markdown('<div class="page-title">🔮 Prediction & Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{TICKERS[ticker]} · {ticker}</div>', unsafe_allow_html=True)

    with st.spinner("Mengambil data..."):
        df = fetch_data(ticker, str(start_date), str(end_date))

    if df.empty:
        st.error("Gagal mengambil data.")
        st.stop()

    meta   = load_meta()
    tmeta  = meta.get('tickers', {}).get(ticker, {})
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    chg    = (latest['Close'] - prev['Close']) / prev['Close'] * 100

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    for col, lbl, val, cls in zip([k1,k2,k3,k4],
        ['Harga Terakhir','Perubahan','Volume','MA7'],
        [f"Rp {latest['Close']:,.0f}", f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%",
         f"{latest['Volume']/1e6:.1f}M", f"Rp {latest['MA7']:,.0f}"],
        ['', 'up' if chg>=0 else 'down', 'muted', 'muted']):
        with col:
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value {cls}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Form Input & Proses</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Parameter yang digunakan untuk prediksi:**")
        feat_df = pd.DataFrame({
            'Fitur': FEATURES,
            'Nilai Terakhir': [f"{latest[f]:.4f}" if f in df.columns else '—' for f in FEATURES]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Konfigurasi Model**")
        st.markdown(f"""<div class="info-box">
            <h4>Best Model</h4>
            <p>{tmeta.get('best_pred_name', 'Linear Regression')}</p>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="info-box">
            <h4>Jumlah Estimator</h4>
            <p>100 (RF & XGBoost)</p>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="info-box">
            <h4>Train/Test Split</h4>
            <p>80% / 20%</p>
        </div>""", unsafe_allow_html=True)

    if not has_model:
        st.info("⬆ Upload model .pkl untuk melihat hasil prediksi ML. Saat ini dalam mode demo.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='#1a1a1a', width=1.5)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'], name='MA7', line=dict(color='#e67e22', width=1, dash='dot')))
        fig.update_layout(**plain_chart(yaxis_title='Harga (Rp)', title='Harga Aktual (Mode Demo)'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        pm, sc_X, sc_y, iso, sc_an = load_models(ticker)
        df['Predicted'] = predict(df, pm, sc_X, sc_y)
        split = int(len(df) * 0.8)
        train, test = df.iloc[:split], df.iloc[split:]

        # Forecast
        row = df.iloc[-1][FEATURES].values.reshape(1,-1).copy()
        last_dt = pd.to_datetime(df['Date'].iloc[-1])
        fc = []
        for i in range(1, forecast_days+1):
            p = sc_y.inverse_transform(pm.predict(sc_X.transform(row)).reshape(-1,1)).ravel()[0]
            fc.append({'Tanggal': last_dt + timedelta(days=i), 'Prediksi Harga (Rp)': p})
            row[0][3] = p
        fc_df = pd.DataFrame(fc)

        st.markdown('<div class="section-label">Hasil Prediksi vs Aktual</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['Date'], y=train['Close'], name='Aktual (Train)', line=dict(color='#cccccc', width=1)))
        fig.add_trace(go.Scatter(x=test['Date'],  y=test['Close'],  name='Aktual (Test)',  line=dict(color='#1a1a1a', width=1.8)))
        fig.add_trace(go.Scatter(x=test['Date'],  y=test['Predicted'], name='Prediksi',    line=dict(color='#e74c3c', width=1.4, dash='dash')))
        fig.add_trace(go.Scatter(x=fc_df['Tanggal'], y=fc_df['Prediksi Harga (Rp)'],
                                  name=f'Forecast {forecast_days}h',
                                  line=dict(color='#27ae60', width=1.5, dash='dot'),
                                  mode='lines+markers', marker=dict(size=4)))
        fig.add_vline(x=str(df.iloc[split]['Date']), line_dash='dash', line_color='#aaa', line_width=1)
        fig.update_layout(**plain_chart(yaxis_title='Harga (Rp)'))
        st.plotly_chart(fig, use_container_width=True)

        # Hasil anomali
        st.markdown('<div class="section-label">Hasil Deteksi Anomali</div>', unsafe_allow_html=True)
        zf, isof, cf = detect_anomaly(df, iso, sc_an)
        df['Anomaly'] = cf
        normal  = df[df['Anomaly'] == 0]
        anomaly = df[df['Anomaly'] == 1]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=normal['Date'], y=normal['Close'], name='Normal',
                                   mode='lines', line=dict(color='#1a1a1a', width=1.4)))
        fig2.add_trace(go.Scatter(x=anomaly['Date'], y=anomaly['Close'], name='Anomali',
                                   mode='markers', marker=dict(color='#e74c3c', size=7)))
        fig2.update_layout(**plain_chart(yaxis_title='Harga (Rp)', title='Deteksi Anomali Konsensus'))
        st.plotly_chart(fig2, use_container_width=True)

        # Tabel forecast
        st.markdown('<div class="section-label">Tabel Forecast</div>', unsafe_allow_html=True)
        fc_disp = fc_df.copy()
        fc_disp['Tanggal'] = fc_disp['Tanggal'].dt.strftime('%d %b %Y')
        fc_disp['Prediksi Harga (Rp)'] = fc_disp['Prediksi Harga (Rp)'].map(lambda x: f"Rp {x:,.0f}")
        st.dataframe(fc_disp, use_container_width=True, hide_index=True)

        # Performa
        st.markdown('<div class="section-label">Performa Model</div>', unsafe_allow_html=True)
        pm1, pm2, pm3, pm4 = st.columns(4)
        for col, lbl, val in zip([pm1,pm2,pm3,pm4],
            ['Best Model','R²','MAPE','MAE'],
            [tmeta.get('best_pred_name','—'), f"{tmeta.get('r2',0):.4f}",
             f"{tmeta.get('mape',0):.2f}%", f"Rp {tmeta.get('mae',0):,.2f}"]):
            with col:
                st.markdown(f"""<div class="kpi-box">
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-value muted">{val}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Visualization":
    st.markdown('<div class="page-title"> Visualization</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{TICKERS[ticker]} · {ticker}</div>', unsafe_allow_html=True)

    with st.spinner("Mengambil data..."):
        df = fetch_data(ticker, str(start_date), str(end_date))

    if df.empty:
        st.error("Gagal mengambil data.")
        st.stop()

    # Candlestick
    st.markdown('<div class="section-label">Grafik Candlestick</div>', unsafe_allow_html=True)
    fig_c = go.Figure(data=[go.Candlestick(
        x=df['Date'].tail(90), open=df['Open'].tail(90),
        high=df['High'].tail(90), low=df['Low'].tail(90), close=df['Close'].tail(90),
        increasing_line_color='#27ae60', decreasing_line_color='#e74c3c'
    )])
    fig_c.update_layout(**plain_chart(title='90 Hari Terakhir — Candlestick', yaxis_title='Harga (Rp)'))
    st.plotly_chart(fig_c, use_container_width=True)

    # Return harian
    st.markdown('<div class="section-label">Distribusi Return Harian</div>', unsafe_allow_html=True)
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(x=df['Daily_Return']*100, nbinsx=60,
                                  marker_color='#1a1a1a', opacity=0.75, name='Return Harian (%)'))
    fig_r.update_layout(**plain_chart(height=300, xaxis_title='Return (%)', yaxis_title='Frekuensi',
                                       title='Distribusi Return Harian'))
    st.plotly_chart(fig_r, use_container_width=True)

    # Volatilitas
    st.markdown('<div class="section-label">Volatilitas (Rolling 7 Hari)</div>', unsafe_allow_html=True)
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=df['Date'], y=df['Volatility']*100,
                                line=dict(color='#8e44ad', width=1.2), name='Volatilitas (%)'))
    fig_v.update_layout(**plain_chart(height=280, yaxis_title='Volatilitas (%)', title='Volatilitas Harga'))
    st.plotly_chart(fig_v, use_container_width=True)

    if has_model:
        pm, sc_X, sc_y, iso, sc_an = load_models(ticker)
        df['Predicted'] = predict(df, pm, sc_X, sc_y)
        split = int(len(df) * 0.8)
        test  = df.iloc[split:]

        st.markdown('<div class="section-label">Visualisasi Hasil Analisis — Prediksi vs Aktual</div>', unsafe_allow_html=True)
        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(x=test['Date'], y=test['Close'],     name='Aktual',   line=dict(color='#1a1a1a', width=2)))
        fig_pa.add_trace(go.Scatter(x=test['Date'], y=test['Predicted'], name='Prediksi', line=dict(color='#e74c3c', width=1.5, dash='dash')))
        fig_pa.update_layout(**plain_chart(yaxis_title='Harga (Rp)', title='Prediksi vs Aktual (Data Test)'))
        st.plotly_chart(fig_pa, use_container_width=True)

        # Scatter aktual vs prediksi
        st.markdown('<div class="section-label">Scatter: Aktual vs Prediksi</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=test['Close'], y=test['Predicted'], mode='markers',
                                     marker=dict(color='#1a1a1a', size=4, opacity=0.5), name='Data'))
        mn = min(test['Close'].min(), test['Predicted'].min())
        mx = max(test['Close'].max(), test['Predicted'].max())
        fig_sc.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                     line=dict(color='#e74c3c', dash='dash'), name='Ideal'))
        fig_sc.update_layout(**plain_chart(height=380, xaxis_title='Aktual (Rp)', yaxis_title='Prediksi (Rp)',
                                            title='Scatter: Aktual vs Prediksi'))
        st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ About":
    st.markdown('<div class="page-title">ℹ About</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Informasi proyek, metode, dan dataset</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Penjelasan Metode</div>', unsafe_allow_html=True)

    metode = [
        ("Linear Regression", "Prediksi", "Model regresi linier untuk memprediksi harga penutupan berdasarkan fitur historis. Sederhana namun efektif sebagai baseline."),
        ("Random Forest", "Prediksi", "Ensemble learning berbasis decision tree. Robust terhadap overfitting dan mampu menangkap hubungan non-linear antar fitur."),
        ("XGBoost", "Prediksi", "Gradient boosting yang dioptimalkan. Performa tinggi dan efisien untuk data tabular dengan banyak fitur."),
        ("SVR (Support Vector Regression)", "Prediksi", "Regresi berbasis support vector dengan kernel RBF. Efektif untuk data dengan pola kompleks dan noise."),
        ("Z-Score", "Anomali", "Mendeteksi anomali berdasarkan simpangan standar. Titik dengan Z-score > 3 dianggap anomali."),
        ("Isolation Forest", "Anomali", "Algoritma berbasis tree yang mengisolasi titik-titik outlier. Efektif untuk data multidimensi."),
        ("Konsensus (≥2 metode)", "Anomali", "Anomali dikonfirmasi jika terdeteksi oleh minimal 2 dari 3 metode deteksi untuk mengurangi false positive."),
    ]
    for m in metode:
        tag_color = "#1a7a3c" if m[1] == "Prediksi" else "#c0392b"
        st.markdown(f"""<div class="method-card">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                <div class="method-title">{m[0]}</div>
                <span style="font-size:0.68rem;background:#f0f0f0;color:{tag_color};padding:2px 8px;border-radius:4px;font-weight:600">{m[1]}</span>
            </div>
            <div class="method-desc">{m[2]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""<div class="info-box">
            <h4>Sumber Data</h4>
            <p>Yahoo Finance melalui library <strong>yfinance</strong>. Data diunduh secara otomatis setiap sesi.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            <h4>Periode</h4>
            <p>1 Januari 2020 — 2 Juni 2026 (± 1.500 hari trading per saham)</p>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""<div class="info-box">
            <h4>Fitur yang Digunakan</h4>
            <p>Open, High, Low, Close, Volume, MA7, MA30, Daily Return, Volatility, Volume Change, High-Low Gap, Open-Close Gap</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            <h4>Target Prediksi</h4>
            <p>Harga Close hari berikutnya (<code>Close.shift(-1)</code>)</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Informasi Proyek</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        <h4>Mata Kuliah</h4>
        <p>Data Mining — Ujian Akhir Semester (UAS)</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Identitas Anggota</div>', unsafe_allow_html=True)
    cols = st.columns(len(TEAM_MEMBERS))
    for col, m in zip(cols, TEAM_MEMBERS):
        with col:
            st.markdown(f"""<div class="member-card">
                <div style="font-size:2rem">👤</div>
                <div class="member-name">{m['name']}</div>
                <div class="member-role">{m['role']}</div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Data: Yahoo Finance · Model: scikit-learn & XGBoost · Bukan saran investasi")
