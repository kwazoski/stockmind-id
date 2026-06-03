import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Saham Indonesia",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: clean, plain, force light theme throughout ──────────────────────────
st.markdown("""
<style>
  /* ── Force light theme globally ── */
  html, body, [class*="css"], [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main, .block-container {
    font-family: 'Segoe UI', Tahoma, Geneva, sans-serif !important;
    font-size: 14px !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background-color: #f5f5f5 !important;
    border-right: 1px solid #e0e0e0 !important;
  }
  [data-testid="stSidebar"] * {
    color: #1a1a1a !important;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] small {
    color: #333333 !important;
  }

  /* ── Selectbox & inputs ── */
  .stSelectbox > div > div,
  [data-testid="stSelectbox"] > div {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid #cccccc !important;
  }
  .stSelectbox svg { fill: #333 !important; }

  /* ── Slider ── */
  .stSlider p, .stSlider span, .stSlider label { color: #333 !important; }
  [data-testid="stSlider"] * { color: #333 !important; }

  /* ── Date inputs ── */
  .stDateInput input {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 1px solid #ccc !important;
  }

  /* ── Tabs ── */
  [data-testid="stTabs"] button {
    color: #555 !important;
    background: transparent !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #1a1a1a !important;
    border-bottom: 2px solid #1a1a1a !important;
  }

  /* ── Divider ── */
  hr { border-color: #e0e0e0 !important; }

  /* ── Caption / small text ── */
  .stCaption, caption, small, [data-testid="stCaptionContainer"] {
    color: #777777 !important;
  }

  /* ── Success / warning alerts ── */
  [data-testid="stAlert"] { border-radius: 6px !important; }

  /* ── KPI cards ── */
  .kpi-box {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 16px 20px;
    background: #fafafa;
    margin-bottom: 4px;
  }
  .kpi-label {
    font-size: 0.72rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
    font-weight: 500;
  }
  .kpi-value { font-size: 1.5rem; font-weight: 700; color: #0a0a0a; }
  .kpi-value.up    { color: #1a7a3c; }
  .kpi-value.down  { color: #c0392b; }
  .kpi-value.muted { color: #222; font-size: 1.1rem; font-weight: 600; }

  /* ── Section labels ── */
  .section-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 8px;
    margin: 24px 0 14px 0;
  }

  /* ── Info / warn notes ── */
  .info-note {
    background: #f0f5ff;
    border-left: 3px solid #4a7cfc;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #333;
    border-radius: 0 4px 4px 0;
    margin-bottom: 14px;
  }

  /* ── Page title ── */
  .page-title { font-size: 2rem; font-weight: 800; color: #0a0a0a; margin-bottom: 4px; letter-spacing: -0.02em; }
  .page-sub   { font-size: 0.85rem; color: #777; margin-bottom: 1.6rem; }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] * { color: #1a1a1a !important; }

  /* ── Hide Streamlit footer ── */
  footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
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

# ── Helpers ──────────────────────────────────────────────────────────────────
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
    defaults = dict(
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family='Segoe UI', size=11, color='#444'),
        legend=dict(orientation='h', y=1.05, x=0),
        xaxis=dict(gridcolor='#f0f0f0', linecolor='#ddd', showgrid=True),
        yaxis=dict(gridcolor='#f0f0f0', linecolor='#ddd', showgrid=True),
    )
    defaults.update(kwargs)
    return defaults

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Prediksi Saham IDX**")
    st.caption("Machine Learning · 5 Emiten")
    st.divider()

    ticker = st.selectbox("Saham", list(TICKERS.keys()),
                          format_func=lambda x: f"{TICKERS[x]} ({x})")
    st.caption("Rentang data")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Dari", value=datetime(2020,1,1), label_visibility="collapsed")
    with col_b:
        end_date   = st.date_input("Sampai", value=datetime.today(), label_visibility="collapsed")

    forecast_days = st.slider("Forecast (hari)", 1, 30, 7)
    st.divider()

    has_model = model_ok(ticker)
    if has_model:
        st.success("Model tersedia", icon="")
    else:
        st.warning("Model belum diupload", icon="")
        st.caption("Upload folder `model/` ke repo GitHub lalu redeploy.")

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f'<div class="page-title">Prediksi Saham Indonesia</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-sub">{TICKERS[ticker]} · {ticker}</div>', unsafe_allow_html=True)

# ── Fetch data ───────────────────────────────────────────────────────────────
with st.spinner("Mengambil data..."):
    df = fetch_data(ticker, str(start_date), str(end_date))

if df.empty:
    st.error("Gagal mengambil data. Coba ubah rentang tanggal.")
    st.stop()

meta       = load_meta()
tmeta      = meta.get('tickers', {}).get(ticker, {})
latest     = df.iloc[-1]
prev       = df.iloc[-2]
chg_pct    = (latest['Close'] - prev['Close']) / prev['Close'] * 100
chg_dir    = "up" if chg_pct >= 0 else "down"
chg_sym    = "▲" if chg_pct >= 0 else "▼"

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("Harga Terakhir", f"Rp {latest['Close']:,.0f}", ""),
    ("Perubahan", f"{chg_sym} {abs(chg_pct):.2f}%", chg_dir),
    ("Volume", f"{latest['Volume']/1e6:.1f}M", "muted"),
    ("MA7", f"Rp {latest['MA7']:,.0f}", "muted"),
    ("R² (Best Model)", f"{tmeta.get('r2', 0):.4f}", "muted"),
]
for col, (label, val, cls) in zip([k1,k2,k3,k4,k5], kpis):
    with col:
        st.markdown(f"""<div class="kpi-box">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediksi Harga", "Deteksi Anomali", "Data & Metrik"])

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Grafik Harga & Prediksi</div>', unsafe_allow_html=True)

    if not has_model:
        st.markdown('<div class="info-note">Mode Demo — model .pkl belum diupload. Menampilkan harga aktual dan moving average.</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close',
                                  line=dict(color='#1a1a1a', width=1.5)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'], name='MA7',
                                  line=dict(color='#e67e22', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA30'], name='MA30',
                                  line=dict(color='#2980b9', width=1, dash='dash')))
        fig.update_layout(**plain_chart(yaxis_title='Harga (Rp)'))
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
            fc.append({'Tanggal': last_dt + timedelta(days=i), 'Harga': p})
            row[0][3] = p
        fc_df = pd.DataFrame(fc)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['Date'], y=train['Close'], name='Aktual (Train)',
                                  line=dict(color='#cccccc', width=1)))
        fig.add_trace(go.Scatter(x=test['Date'], y=test['Close'], name='Aktual (Test)',
                                  line=dict(color='#1a1a1a', width=1.8)))
        fig.add_trace(go.Scatter(x=test['Date'], y=test['Predicted'], name='Prediksi',
                                  line=dict(color='#e74c3c', width=1.4, dash='dash')))
        fig.add_trace(go.Scatter(x=fc_df['Tanggal'], y=fc_df['Harga'],
                                  name=f'Forecast {forecast_days}h',
                                  line=dict(color='#27ae60', width=1.5, dash='dot'),
                                  mode='lines+markers', marker=dict(size=4)))
        fig.add_vline(x=str(df.iloc[split]['Date']), line_dash='dash',
                      line_color='#aaa', line_width=1)
        fig.update_layout(**plain_chart(yaxis_title='Harga (Rp)'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Tabel Forecast</div>', unsafe_allow_html=True)
        fc_df['Tanggal'] = fc_df['Tanggal'].dt.strftime('%d %b %Y')
        fc_df['Harga']   = fc_df['Harga'].map(lambda x: f"Rp {x:,.0f}")
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-label">Performa Model</div>', unsafe_allow_html=True)
        pm1, pm2, pm3, pm4 = st.columns(4)
        for col, lbl, val in zip(
            [pm1, pm2, pm3, pm4],
            ['Best Model', 'R²', 'MAPE', 'MAE'],
            [tmeta.get('best_pred_model','—'),
             f"{tmeta.get('r2',0):.4f}",
             f"{tmeta.get('mape',0):.2f}%",
             f"Rp {tmeta.get('mae',0):,.2f}"]
        ):
            with col:
                st.markdown(f"""<div class="kpi-box">
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-value muted">{val}</div>
                </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Deteksi Anomali Harga</div>', unsafe_allow_html=True)

    if not has_model:
        st.markdown('<div class="info-note">Mode Demo — menggunakan Z-Score sederhana.</div>', unsafe_allow_html=True)
        cz = np.abs(stats.zscore(df['Close'].values))
        df['Anomaly'] = (cz > 2.5).astype(int)
    else:
        pm, sc_X, sc_y, iso, sc_an = load_models(ticker)
        zf, isof, cf = detect_anomaly(df, iso, sc_an)
        df['Anomaly_Z'] = zf
        df['Anomaly_IF'] = isof
        df['Anomaly'] = cf

    normal  = df[df['Anomaly'] == 0]
    anomaly = df[df['Anomaly'] == 1]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=normal['Date'], y=normal['Close'], name='Normal',
                               mode='lines', line=dict(color='#1a1a1a', width=1.4)))
    fig2.add_trace(go.Scatter(x=anomaly['Date'], y=anomaly['Close'], name='Anomali',
                               mode='markers', marker=dict(color='#e74c3c', size=7,
                               line=dict(color='#c0392b', width=1))))
    fig2.update_layout(**plain_chart(yaxis_title='Harga (Rp)'))
    st.plotly_chart(fig2, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    for col, lbl, val in zip(
        [a1, a2, a3],
        ['Total Anomali', 'Persentase', 'Total Hari Data'],
        [f"{len(anomaly)}", f"{len(anomaly)/len(df)*100:.1f}%", f"{len(df):,}"]
    ):
        with col:
            st.markdown(f"""<div class="kpi-box">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value muted">{val}</div>
            </div>""", unsafe_allow_html=True)

    if has_model:
        st.markdown('<div class="section-label">Breakdown per Metode</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        for col, lbl, val in zip(
            [b1, b2, b3],
            ['Z-Score', 'Isolation Forest', 'Konsensus (≥2)'],
            [df['Anomaly_Z'].sum(), df['Anomaly_IF'].sum(), df['Anomaly'].sum()]
        ):
            with col:
                st.markdown(f"""<div class="kpi-box">
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-value muted">{val}</div>
                </div>""", unsafe_allow_html=True)

    if not anomaly.empty:
        st.markdown('<div class="section-label">Tanggal Anomali Terbaru</div>', unsafe_allow_html=True)
        show = anomaly[['Date','Close','Volume','Daily_Return','Volatility']].tail(15).copy()
        show['Date']         = pd.to_datetime(show['Date']).dt.strftime('%d %b %Y')
        show['Close']        = show['Close'].map(lambda x: f"Rp {x:,.0f}")
        show['Volume']       = show['Volume'].map(lambda x: f"{x/1e6:.1f}M")
        show['Daily_Return'] = show['Daily_Return'].map(lambda x: f"{x*100:.2f}%")
        show['Volatility']   = show['Volatility'].map(lambda x: f"{x*100:.2f}%")
        show.columns = ['Tanggal','Close','Volume','Return Harian','Volatilitas']
        st.dataframe(show, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Data Harga (30 hari terakhir)</div>', unsafe_allow_html=True)
    disp = df[['Date','Open','High','Low','Close','Volume','MA7','MA30','Daily_Return']].tail(30).copy()
    disp['Date']         = pd.to_datetime(disp['Date']).dt.strftime('%d %b %Y')
    disp['Daily_Return'] = disp['Daily_Return'].map(lambda x: f"{x*100:.2f}%")
    for c in ['Open','High','Low','Close','MA7','MA30']:
        disp[c] = disp[c].map(lambda x: f"Rp {x:,.0f}")
    disp['Volume'] = disp['Volume'].map(lambda x: f"{x/1e6:.1f}M")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">Statistik Deskriptif</div>', unsafe_allow_html=True)
    st.dataframe(df[['Close','Volume','Daily_Return','Volatility']].describe().round(4),
                 use_container_width=True)

    if meta.get('tickers'):
        st.markdown('<div class="section-label">Ringkasan Semua Saham</div>', unsafe_allow_html=True)
        rows = []
        for t, info in meta['tickers'].items():
            rows.append({
                'Saham': t, 'Perusahaan': info.get('company','—'),
                'Best Model': info.get('best_pred_model','—'),
                'R²': f"{info.get('r2',0):.4f}",
                'MAPE': f"{info.get('mape',0):.2f}%",
                'Anomali': info.get('n_anomaly','—'),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("Data: Yahoo Finance · Model: scikit-learn & XGBoost · Bukan saran investasi")
