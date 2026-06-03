import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockMind ID — Prediksi Saham Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }
  .stApp {
    background: #0a0e1a;
    color: #e8eaf0;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #00d4ff, #7b2ff7, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
  }
  .hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #5a6480;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }
  .metric-card {
    background: #12182e;
    border: 1px solid #1e2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #4a5580;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #00d4ff;
  }
  .metric-value.green { color: #00e676; }
  .metric-value.red   { color: #ff4444; }
  .metric-value.gold  { color: #ffd740; }
  .badge {
    display: inline-block;
    background: #1e2a4a;
    border: 1px solid #2a3a6a;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #7b8ec0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .badge.best {
    background: linear-gradient(135deg, #1a2a1a, #1a3a1a);
    border-color: #00e676;
    color: #00e676;
  }
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4a5580;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    border-bottom: 1px solid #1e2a4a;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
  }
  .anomaly-tag {
    background: #2a1a1a;
    border: 1px solid #ff4444;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #ff6666;
  }
  [data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e2a4a;
  }
  .stSelectbox label, .stSlider label, .stDateInput label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #5a6480;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  div[data-testid="stMetric"] {
    background: #12182e;
    border: 1px solid #1e2a4a;
    border-radius: 10px;
    padding: 1rem;
  }
  .stAlert { border-radius: 10px; }
  h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; color: #c8d0e8; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
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

# ─── Helper Functions ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.reset_index(inplace=True)
    df['MA7']           = df['Close'].rolling(7).mean()
    df['MA30']          = df['Close'].rolling(30).mean()
    df['Daily_Return']  = df['Close'].pct_change()
    df['Volatility']    = df['Daily_Return'].rolling(7).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['High_Low_Gap']  = df['High'] - df['Low']
    df['Open_Close_Gap']= df['Close'] - df['Open']
    df['Target']        = df['Close'].shift(-1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_models(ticker: str):
    ticker_clean = ticker.replace('.', '_')
    folder = os.path.join(MODEL_DIR, ticker_clean)
    pred_model  = joblib.load(os.path.join(folder, 'model_prediction.pkl'))
    sc_X        = joblib.load(os.path.join(folder, 'scaler_X.pkl'))
    sc_y        = joblib.load(os.path.join(folder, 'scaler_y.pkl'))
    iso_model   = joblib.load(os.path.join(folder, 'model_anomaly.pkl'))
    sc_an       = joblib.load(os.path.join(folder, 'scaler_anomaly.pkl'))
    return pred_model, sc_X, sc_y, iso_model, sc_an


def load_metadata():
    path = os.path.join(MODEL_DIR, 'metadata.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run_prediction(df, pred_model, sc_X, sc_y):
    X = df[FEATURES].values
    X_sc = sc_X.transform(X)
    y_pred_sc = pred_model.predict(X_sc).reshape(-1,1)
    y_pred = sc_y.inverse_transform(y_pred_sc).ravel()
    return y_pred


def run_anomaly(df, iso_model, sc_an):
    X_an = df[ANOMALY_FEATURES].copy()
    X_sc = sc_an.transform(X_an)
    z = np.abs(stats.zscore(X_sc))
    zscore_flag    = (z > 3).any(axis=1).astype(int)
    iso_flag       = (iso_model.predict(X_sc) == -1).astype(int)
    consensus_flag = ((zscore_flag + iso_flag) >= 2).astype(int)
    return zscore_flag, iso_flag, consensus_flag


def model_has_files(ticker: str) -> bool:
    ticker_clean = ticker.replace('.', '_')
    folder = os.path.join(MODEL_DIR, ticker_clean)
    needed = ['model_prediction.pkl','scaler_X.pkl','scaler_y.pkl',
              'model_anomaly.pkl','scaler_anomaly.pkl']
    return all(os.path.exists(os.path.join(folder, f)) for f in needed)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">StockMind<br>ID</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ML · IDX Stocks</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pilih Saham</div>', unsafe_allow_html=True)
    ticker_selected = st.selectbox(
        "Saham",
        options=list(TICKERS.keys()),
        format_func=lambda x: f"{TICKERS[x]} ({x})",
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-header">Rentang Waktu</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Mulai", value=datetime(2020,1,1), label_visibility="collapsed")
    with col2:
        end_date   = st.date_input("Akhir", value=datetime.today(), label_visibility="collapsed")

    st.markdown('<div class="section-header">Prediksi ke Depan</div>', unsafe_allow_html=True)
    forecast_days = st.slider("Hari ke depan", 1, 30, 7, label_visibility="collapsed")

    st.markdown("---")
    models_available = model_has_files(ticker_selected)
    if models_available:
        st.success("✅ Model tersedia")
    else:
        st.warning("⚠️ Model .pkl belum diupload")
        st.markdown("""
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#5a6480;line-height:1.7'>
        Upload model dari Colab:<br>
        1. Extract <code>models_saham_indonesia.zip</code><br>
        2. Letakkan folder <code>model/</code> di root project<br>
        3. Re-deploy ke Streamlit Cloud
        </div>
        """, unsafe_allow_html=True)


# ─── MAIN ────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="hero-title">📈 StockMind Indonesia</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Prediksi Harga · Deteksi Anomali · 5 Saham IDX</div>', unsafe_allow_html=True)

company   = TICKERS[ticker_selected]
meta      = load_metadata()
ticker_meta = meta.get('tickers', {}).get(ticker_selected, {})

# ─── Load & process data ─────────────────────────────────────────────────────
with st.spinner(f"Mengambil data {company}..."):
    df = fetch_data(ticker_selected, str(start_date), str(end_date))

if df.empty:
    st.error("Gagal mengambil data. Coba ubah rentang tanggal.")
    st.stop()

# ─── TOP METRICS ─────────────────────────────────────────────────────────────
latest      = df.iloc[-1]
prev        = df.iloc[-2]
price_chg   = latest['Close'] - prev['Close']
price_pct   = price_chg / prev['Close'] * 100
chg_color   = "green" if price_chg >= 0 else "red"
chg_arrow   = "▲" if price_chg >= 0 else "▼"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Harga Terakhir</div>
        <div class="metric-value">Rp {latest['Close']:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Perubahan</div>
        <div class="metric-value {chg_color}">{chg_arrow} {abs(price_pct):.2f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Volume</div>
        <div class="metric-value" style="font-size:1.2rem">{latest['Volume']/1e6:.1f}M</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MA7</div>
        <div class="metric-value" style="font-size:1.3rem">Rp {latest['MA7']:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    best_model_name = ticker_meta.get('best_pred_model', 'Linear Regression')
    r2_val = ticker_meta.get('r2', 0)
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Best Model R²</div>
        <div class="metric-value gold">{r2_val:.4f}</div>
    </div>""", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Prediksi Harga", "🔍 Deteksi Anomali", "📋 Data & Metrik"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Prediksi vs Aktual + Forecast</div>', unsafe_allow_html=True)

    if not models_available:
        # ── Demo mode: show actual price + moving averages only ──
        st.info("🔄 Mode Demo — Upload model .pkl untuk prediksi ML. Menampilkan harga aktual + MA.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Harga Close',
                                  line=dict(color='#00d4ff', width=1.5)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'], name='MA7',
                                  line=dict(color='#ffd740', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA30'], name='MA30',
                                  line=dict(color='#ff6b35', width=1, dash='dash')))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1220',
            height=480, margin=dict(l=10,r=10,t=30,b=10),
            legend=dict(orientation='h', y=1.02, x=0),
            xaxis=dict(gridcolor='#1e2a4a'), yaxis=dict(gridcolor='#1e2a4a'),
            font=dict(family='Space Mono', size=11)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        pred_model, sc_X, sc_y, iso_model, sc_an = load_models(ticker_selected)
        y_pred = run_prediction(df, pred_model, sc_X, sc_y)
        df['Predicted'] = y_pred

        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx]
        df_test  = df.iloc[split_idx:]

        # ── Forecast next N days ──
        last_row = df.iloc[-1][FEATURES].values.reshape(1,-1)
        forecasts = []
        row = last_row.copy()
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        for i in range(1, forecast_days+1):
            pred_sc = pred_model.predict(sc_X.transform(row)).reshape(-1,1)
            pred_price = sc_y.inverse_transform(pred_sc).ravel()[0]
            forecasts.append({'date': last_date + timedelta(days=i), 'price': pred_price})
            # shift close in row
            row[0][3] = pred_price  # update Close feature approx

        forecast_df = pd.DataFrame(forecasts)

        # ── Main chart ──
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Close'],
                                  name='Aktual (Train)', line=dict(color='#2a3a6a', width=1)))
        fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Close'],
                                  name='Aktual (Test)', line=dict(color='#00d4ff', width=2)))
        fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Predicted'],
                                  name='Prediksi', line=dict(color='#ff6b35', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['price'],
                                  name=f'Forecast {forecast_days}h',
                                  line=dict(color='#00e676', width=2, dash='dot'),
                                  mode='lines+markers', marker=dict(size=5)))
        # Vertical separator
        fig.add_vline(x=str(df.iloc[split_idx]['Date']), line_dash='dash',
                      line_color='#4a5580', line_width=1,
                      annotation_text="Train|Test", annotation_font_color="#4a5580")

        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1220',
            height=500, margin=dict(l=10,r=10,t=30,b=10),
            legend=dict(orientation='h', y=1.02, x=0),
            xaxis=dict(gridcolor='#1e2a4a'), yaxis=dict(gridcolor='#1e2a4a', title='Harga (Rp)'),
            font=dict(family='Space Mono', size=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Forecast table ──
        st.markdown('<div class="section-header">Tabel Forecast</div>', unsafe_allow_html=True)
        forecast_df['date'] = forecast_df['date'].dt.strftime('%d %b %Y')
        forecast_df.columns = ['Tanggal', 'Prediksi Harga (Rp)']
        forecast_df['Prediksi Harga (Rp)'] = forecast_df['Prediksi Harga (Rp)'].map(lambda x: f"Rp {x:,.0f}")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # ── Model performance ──
        st.markdown('<div class="section-header">Performa Model (dari Training)</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        perf_cols = [m1, m2, m3, m4]
        labels = ['Best Model','R²','MAPE','MAE']
        values = [
            ticker_meta.get('best_pred_model','—'),
            f"{ticker_meta.get('r2',0):.4f}",
            f"{ticker_meta.get('mape',0):.2f}%",
            f"Rp {ticker_meta.get('mae',0):,.2f}",
        ]
        for col, lbl, val in zip(perf_cols, labels, values):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value gold" style="font-size:1.1rem">{val}</div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Deteksi Anomali Harga</div>', unsafe_allow_html=True)

    if not models_available:
        st.info("🔄 Mode Demo — Upload model .pkl untuk anomali berbasis ML.")
        # Z-Score only demo
        close_z = np.abs(stats.zscore(df['Close'].values))
        df['Anomaly_Demo'] = (close_z > 2.5).astype(int)
        normal  = df[df['Anomaly_Demo'] == 0]
        anomaly = df[df['Anomaly_Demo'] == 1]
    else:
        pred_model, sc_X, sc_y, iso_model, sc_an = load_models(ticker_selected)
        zscore_f, iso_f, consensus_f = run_anomaly(df, iso_model, sc_an)
        df['Anomaly_ZScore']    = zscore_f
        df['Anomaly_IsoForest'] = iso_f
        df['Anomaly_Consensus'] = consensus_f
        normal  = df[df['Anomaly_Consensus'] == 0]
        anomaly = df[df['Anomaly_Consensus'] == 1]

    an_col = 'Anomaly_Consensus' if 'Anomaly_Consensus' in df.columns else 'Anomaly_Demo'

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=normal['Date'], y=normal['Close'],
                               name='Normal', mode='lines',
                               line=dict(color='#00d4ff', width=1.5)))
    fig2.add_trace(go.Scatter(x=anomaly['Date'], y=anomaly['Close'],
                               name='Anomali', mode='markers',
                               marker=dict(color='#ff4444', size=8, symbol='circle',
                                           line=dict(color='#ff8888', width=1))))
    fig2.update_layout(
        template='plotly_dark', paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1220',
        height=480, margin=dict(l=10,r=10,t=30,b=10),
        legend=dict(orientation='h', y=1.02, x=0),
        xaxis=dict(gridcolor='#1e2a4a'), yaxis=dict(gridcolor='#1e2a4a', title='Harga (Rp)'),
        font=dict(family='Space Mono', size=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Stats
    n_anomaly = len(anomaly)
    pct_anomaly = n_anomaly / len(df) * 100
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Anomali</div>
            <div class="metric-value red">{n_anomaly} titik</div>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Persentase</div>
            <div class="metric-value red">{pct_anomaly:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with cc:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Hari</div>
            <div class="metric-value">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)

    if models_available:
        st.markdown('<div class="section-header">Breakdown per Metode</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Z-Score</div>
                <div class="metric-value red">{df['Anomaly_ZScore'].sum()}</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Isolation Forest</div>
                <div class="metric-value red">{df['Anomaly_IsoForest'].sum()}</div>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Konsensus (≥2)</div>
                <div class="metric-value red">{df['Anomaly_Consensus'].sum()}</div>
            </div>""", unsafe_allow_html=True)

    # Table of anomaly dates
    if not anomaly.empty:
        st.markdown('<div class="section-header">Tanggal Anomali Terkini</div>', unsafe_allow_html=True)
        an_display = anomaly[['Date','Close','Volume','Daily_Return','Volatility']].tail(15).copy()
        an_display['Date']         = pd.to_datetime(an_display['Date']).dt.strftime('%d %b %Y')
        an_display['Close']        = an_display['Close'].map(lambda x: f"Rp {x:,.0f}")
        an_display['Volume']       = an_display['Volume'].map(lambda x: f"{x/1e6:.1f}M")
        an_display['Daily_Return'] = an_display['Daily_Return'].map(lambda x: f"{x*100:.2f}%")
        an_display['Volatility']   = an_display['Volatility'].map(lambda x: f"{x*100:.2f}%")
        an_display.columns = ['Tanggal','Close','Volume','Return Harian','Volatilitas']
        st.dataframe(an_display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Data Harga Terbaru</div>', unsafe_allow_html=True)
    disp = df[['Date','Open','High','Low','Close','Volume','MA7','MA30','Daily_Return','Volatility']].tail(30).copy()
    disp['Date']         = pd.to_datetime(disp['Date']).dt.strftime('%d %b %Y')
    disp['Daily_Return'] = disp['Daily_Return'].map(lambda x: f"{x*100:.2f}%")
    disp['Volatility']   = disp['Volatility'].map(lambda x: f"{x*100:.2f}%")
    for col in ['Open','High','Low','Close']:
        disp[col] = disp[col].map(lambda x: f"Rp {x:,.0f}")
    disp['Volume'] = disp['Volume'].map(lambda x: f"{x/1e6:.1f}M")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
    desc = df[['Close','Volume','Daily_Return','Volatility','High_Low_Gap']].describe().round(4)
    st.dataframe(desc, use_container_width=True)

    if meta:
        st.markdown('<div class="section-header">Info Model (metadata.json)</div>', unsafe_allow_html=True)
        tickers_meta = meta.get('tickers', {})
        if tickers_meta:
            meta_rows = []
            for t, info in tickers_meta.items():
                meta_rows.append({
                    'Saham': t,
                    'Perusahaan': info.get('company','—'),
                    'Best Model': info.get('best_pred_model','—'),
                    'R²': f"{info.get('r2',0):.4f}",
                    'MAPE': f"{info.get('mape',0):.2f}%",
                    'Anomali': info.get('n_anomaly','—'),
                })
            st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#2a3a5a;text-align:center;padding:1rem 0'>
  StockMind ID · Data via Yahoo Finance · Model via scikit-learn & XGBoost<br>
  ⚠ Bukan saran investasi — gunakan hanya untuk tujuan akademik/riset
</div>
""", unsafe_allow_html=True)
