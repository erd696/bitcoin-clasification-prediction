import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import re
import os
import yfinance as yf
from vmdpy import VMD

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="BTC Price Direction",
    page_icon="₿",
    layout="centered"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #F7931A;
    margin-bottom: 0;
}
.sub-title {
    font-size: 0.9rem;
    color: #888;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}
.pred-box-up {
    background: linear-gradient(135deg, #0d2b1e, #0a3d20);
    border: 1.5px solid #22c55e;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
    margin-top: 1.5rem;
}
.pred-box-down {
    background: linear-gradient(135deg, #2b0d0d, #3d0a0a);
    border: 1.5px solid #ef4444;
    border-radius: 12px;
    padding: 24px 32px;
    text-align: center;
    margin-top: 1.5rem;
}
.pred-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #aaa;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pred-value-up  { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #22c55e; }
.pred-value-down { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #ef4444; }
.pred-desc { font-size: 0.85rem; color: #aaa; margin-top: 8px; }
.info-box {
    background: #1a1a2e;
    border-left: 3px solid #F7931A;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #ccc;
    margin-bottom: 1rem;
}
.step-badge {
    background: #F7931A;
    color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 8px;
}
.metric-row { display: flex; gap: 12px; margin-top: 1rem; }
.metric-card {
    flex: 1;
    background: #111;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.metric-val { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #F7931A; }
.metric-lbl { font-size: 0.7rem; color: #888; margin-top: 4px; }
.warning-box {
    background: #1a1500;
    border-left: 3px solid #eab308;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #ccc;
    margin-top: 1rem;
}
.hist-box {
    background: #0d1117;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #aaa;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# KONSTANTA MODEL TERBAIK (RF 70:30)
# — Dipilih berdasarkan evaluasi komprehensif:
#   F1=0.6656 (tertinggi seimbang), Acc=0.6519,
#   Prec-Recall gap terkecil (0.0546), stabilitas ±0.0325
# ============================================================
MODEL_NAME  = "Random Forest"
SPLIT       = "TS 70:30"
SAFE_SPLIT  = "TS_70_30"

MODEL_PERF = {
    "Acc":  "0.6519",
    "Prec": "0.6394",
    "Rec":  "0.6940",
    "F1":   "0.6656",
}


# ============================================================
# FUNGSI HELPER
# ============================================================
def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_wilder(close, n=14):
    delta = close.diff()
    gain  = delta.clip(lower=0.0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    loss  = -delta.clip(upper=0.0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def compute_features(df):
    feat = df.copy()
    feat["EMA_10"]        = ema(feat["Close"], 10)
    feat["RSI_14"]        = rsi_wilder(feat["Close"], 14)
    feat["STOCH_14"]      = ((feat["Close"] - feat["Low"].rolling(14).min()) /
                             (feat["High"].rolling(14).max() - feat["Low"].rolling(14).min()))
    feat["PROC"]          = (feat["Close"] - feat["Close"].shift(14)) / feat["Close"].shift(14)
    feat["Std_20"]        = feat["Close"].rolling(20).std()
    feat["Volume_Change"] = feat["Volume"].pct_change()
    return feat

def apply_vmd_global(series, K, alpha=2000, tau=0., DC=0, init=1, tol=1e-7):
    signal = series.values
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    min_len = min(len(series), u.shape[1])
    modes = {f"VMD_{i+1}": u[i][:min_len] for i in range(K)}
    return pd.DataFrame(modes, index=series.index[:min_len])


# ============================================================
# FETCH DATA HISTORIS OTOMATIS
# ============================================================
@st.cache_data(ttl=1800)
def fetch_historical_data():
    """Ambil data historis BTC dari Yahoo Finance dengan retry logic."""
    import time

    last_exc = None
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(8 * attempt)

            START_DATE = "2014-09-17"
            END_DATE = "2025-11-01"

            df = yf.download(
                "BTC-USD",
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if df is None or df.empty:
                last_exc = ValueError(
                    "Yahoo Finance mengembalikan data kosong. "
                    "Kemungkinan rate-limited — coba refresh halaman dalam 1-2 menit."
                )
                continue

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
            if missing:
                raise ValueError(f"Kolom tidak lengkap dari Yahoo Finance: {missing}")

            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            df = df.dropna(how="all").sort_index()

            if df.empty:
                last_exc = ValueError("Data setelah pembersihan masih kosong.")
                continue

            return df

        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ["Rate", "429", "Too Many"]):
                last_exc = ValueError(
                    f"Yahoo Finance rate-limit (percobaan {attempt+1}/3). "
                    "Tunggu sebentar lalu refresh halaman."
                )
                time.sleep(15 * (attempt + 1))
            else:
                last_exc = e

    raise last_exc or RuntimeError("Gagal mengambil data setelah 3 percobaan.")


# ============================================================
# LOAD ARTIFACTS (CACHED)
# ============================================================
@st.cache_resource
def load_metadata():
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    with open(base_dir / "saved_meta" / "feature_meta.json") as f:
        meta = json.load(f)
    with open(base_dir / "saved_meta" / "best_k_vmd.json") as f:
        best_k = json.load(f)
    return meta, best_k

@st.cache_resource
def load_scaler():
    from pathlib import Path
    base_dir   = Path(__file__).resolve().parent
    model_path = base_dir / "saved_scalers" / f"scaler_{SAFE_SPLIT}.joblib"
    return joblib.load(str(model_path))

@st.cache_resource
def load_rf():
    from pathlib import Path
    import traceback
    base_dir   = Path(__file__).resolve().parent
    model_path = base_dir / "saved_models_rf" / f"best_rf_{SAFE_SPLIT}.joblib"
    try:
        return joblib.load(str(model_path))
    except Exception as e:
        st.error(f"❌ Gagal load RF — {type(e).__name__}: {str(e)}")
        st.code(traceback.format_exc())
        return None


# ============================================================
# SIDEBAR — INFO MODEL TERBAIK
# ============================================================
with st.sidebar:
    st.markdown("### 🏆 Model Terbaik")
    st.markdown("---")

    st.markdown(f"""
    <div style='background:#1a1a2e;border-left:3px solid #F7931A;border-radius:6px;
                padding:10px 14px;font-size:0.85rem;color:#ccc;margin-bottom:0.8rem;'>
        🤖 <b>Algoritma:</b> <span style='color:#F7931A'>Random Forest</span><br>
        🎯 <b>Split:</b> <span style='color:#F7931A'>TS 70:30</span>
    </div>
    <div style='font-size:0.75rem;color:#666;margin-bottom:1rem;'>
        Dipilih berdasarkan evaluasi komprehensif: F1 tertinggi di antara model yang seimbang,
        gap Precision-Recall terkecil (0,0546), dan stabilitas antar-skenario terbaik (±0,0325).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Performa Model")
    st.markdown(f"""
    <div style='font-size:0.82rem;color:#aaa;line-height:2.0'>
    Accuracy &nbsp;: <b style='color:#F7931A'>{MODEL_PERF['Acc']}</b><br>
    Precision : <b style='color:#F7931A'>{MODEL_PERF['Prec']}</b><br>
    Recall &nbsp;&nbsp;&nbsp;: <b style='color:#F7931A'>{MODEL_PERF['Rec']}</b><br>
    F1-Score &nbsp;: <b style='color:#F7931A'>{MODEL_PERF['F1']}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Tugas Akhir — Prediksi Arah Harga BTC menggunakan VMD + Random Forest")


# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-title">₿ BTC Price Direction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Prediksi arah harga Bitcoin untuk hari berikutnya</p>', unsafe_allow_html=True)

# Fetch & tampilkan info data historis
with st.spinner("Mengambil data historis BTC dari Yahoo Finance..."):
    try:
        df_hist = fetch_historical_data()
        last_date  = df_hist.index[-1].strftime("%d %b %Y")
        last_close = df_hist["Close"].iloc[-1]
        st.markdown(f"""
        <div class="hist-box">
            ✅ <b>Data historis berhasil diambil otomatis</b> dari Yahoo Finance<br>
            📅 Data terakhir: <b>{last_date}</b> &nbsp;|&nbsp;
            💰 Close terakhir: <b>${last_close:,.2f}</b> &nbsp;|&nbsp;
            📊 Total: <b>{len(df_hist)} hari</b>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        err_msg = str(e)
        st.error(f"❌ Gagal mengambil data historis: {err_msg}")
        if any(k in err_msg for k in ["rate", "Rate", "429", "Too Many", "kosong"]):
            st.warning(
                "⏳ **Yahoo Finance sedang membatasi permintaan (rate limit).**\n\n"
                "Ini umum terjadi di Streamlit Cloud karena banyak app berbagi IP yang sama. "
                "Silakan tunggu **1–2 menit** lalu klik tombol di bawah untuk mencoba lagi."
            )
            if st.button("🔄 Coba Lagi Sekarang"):
                st.cache_data.clear()
                st.rerun()
        else:
            st.info("Pastikan koneksi internet aktif dan coba refresh halaman.")
        st.stop()

st.markdown("""
<div class="info-box">
ℹ️ <b>Cara kerja:</b> Pilih tanggal dari data historis yang sudah lengkap (OHLCV tersedia).
Data akan terisi otomatis, lalu model memprediksi apakah harga BTC akan
<b>naik atau turun</b> pada hari <b>setelah</b> tanggal yang dipilih.
</div>
""", unsafe_allow_html=True)


# ============================================================
# FORM INPUT
# ============================================================
st.markdown("### <span class='step-badge'>01</span> Pilih Tanggal Data", unsafe_allow_html=True)
st.caption("Pilih tanggal dengan data OHLCV lengkap. Model akan memprediksi arah harga hari berikutnya.")

today = pd.Timestamp.now().normalize()
valid_dates = df_hist.index.normalize().unique().sort_values(ascending=False)
valid_dates = valid_dates[valid_dates < today]
valid_dates_list = [d.date() for d in valid_dates]

selected_date = st.date_input(
    "📅 Pilih Tanggal",
    value=valid_dates_list[0],
    min_value=valid_dates_list[-1],
    max_value=(pd.Timestamp.now() - pd.Timedelta(days=1)).date(),
    help="Hanya tanggal dengan data penutupan lengkap yang tersedia",
    format="DD/MM/YYYY"
)

selected_ts = pd.Timestamp(selected_date)
if selected_ts not in df_hist.index:
    available_before = df_hist.index[df_hist.index < selected_ts]
    if available_before.empty:
        st.error("❌ Tidak ada data historis yang tersedia sebelum tanggal yang dipilih.")
        st.stop()
    selected_ts = available_before[-1]
    st.info(
        f"ℹ️ Data untuk **{selected_date.strftime('%d %b %Y')}** belum tersedia di Yahoo Finance. "
        f"Menggunakan data terakhir yang tersedia: **{selected_ts.strftime('%d %b %Y')}**"
    )

row        = df_hist.loc[selected_ts]
open_val   = float(row["Open"])
high_val   = float(row["High"])
low_val    = float(row["Low"])
close_val  = float(row["Close"])
volume_val = float(row["Volume"])

st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.number_input("📈 Open (USD)",   value=open_val,   format="%.2f", disabled=True)
    st.number_input("📈 High (USD)",   value=high_val,   format="%.2f", disabled=True)
    st.number_input("📉 Low (USD)",    value=low_val,    format="%.2f", disabled=True)

with col2:
    st.number_input("💰 Close (USD)",  value=close_val,  format="%.2f", disabled=True)
    st.number_input("📊 Volume (USD)", value=volume_val, format="%.0f", disabled=True)
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.button(
        "🔮 Prediksi Sekarang",
        use_container_width=True,
        type="primary"
    )

next_date_str = (selected_ts + pd.Timedelta(days=1)).strftime("%d %b %Y")
st.caption(f"💡 Data diisi otomatis dari Yahoo Finance. Hasil prediksi = arah harga pada **{next_date_str}**.")


# ============================================================
# PREDIKSI
# ============================================================
if submitted:
    with st.spinner("⏳ Memproses fitur dan menjalankan prediksi..."):
        try:
            # Load metadata
            meta, best_k_dict = load_metadata()
            ohlcv_cols = meta["ohlcv_cols"]
            best_k     = best_k_dict.get(SAFE_SPLIT) or best_k_dict.get(SPLIT)

            df_all = df_hist.loc[:selected_ts].copy()

            feat = compute_features(df_all)
            feat = feat[ohlcv_cols].dropna().copy()

            if len(feat) < 1:
                st.error("❌ Data tidak cukup untuk membuat fitur. Pilih tanggal yang lebih baru.")
                st.stop()

            vmd_df = apply_vmd_global(feat["Close"], K=best_k)
            vmd_df = vmd_df.loc[vmd_df.index.isin(feat.index)].dropna()
            feat   = feat.loc[vmd_df.index].copy()

            X_full   = pd.concat([feat[ohlcv_cols], vmd_df], axis=1)
            scaler   = load_scaler()
            X_scaled = scaler.transform(X_full)

            model = load_rf()
            if model is None:
                st.stop()

            prob       = float(model.predict_proba(X_scaled[[-1]])[0][1])
            pred_label = 1 if prob >= 0.5 else 0

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

    # ============================================================
    # HASIL PREDIKSI
    # ============================================================
    st.markdown("### <span class='step-badge'>02</span> Hasil Prediksi", unsafe_allow_html=True)

    pred_next_date = (selected_ts + pd.Timedelta(days=1)).strftime("%d %b %Y")

    if pred_label == 1:
        st.markdown(f"""
        <div class="pred-box-up">
            <div class="pred-label">PREDIKSI ARAH HARGA — {pred_next_date}</div>
            <div class="pred-value-up">▲ NAIK</div>
            <div class="pred-desc">
                Model memprediksi harga BTC akan <b>naik</b> dari Close
                {selected_ts.strftime("%d %b %Y")} (${close_val:,.2f})
            </div>
            <div style="margin-top:14px;font-family:'Space Mono',monospace;font-size:1rem;color:#86efac;">
                Probabilitas Naik: <b>{prob*100:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="pred-box-down">
            <div class="pred-label">PREDIKSI ARAH HARGA — {pred_next_date}</div>
            <div class="pred-value-down">▼ TURUN</div>
            <div class="pred-desc">
                Model memprediksi harga BTC akan <b>turun</b> dari Close
                {selected_ts.strftime("%d %b %Y")} (${close_val:,.2f})
            </div>
            <div style="margin-top:14px;font-family:'Space Mono',monospace;font-size:1rem;color:#fca5a5;">
                Probabilitas Naik: <b>{prob*100:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">Random Forest</div>
            <div class="metric-lbl">Model</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">TS 70:30</div>
            <div class="metric-lbl">Split Terbaik</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">K={best_k}</div>
            <div class="metric-lbl">VMD Mode</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{len(feat)}</div>
            <div class="metric-lbl">Baris Diproses</div>
        </div>
    </div>
    <div class="warning-box">
        ⚠️ <b>Disclaimer:</b> Prediksi ini hanya untuk keperluan penelitian akademik.
        Bukan merupakan saran investasi. Harga aset kripto sangat volatil.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.75rem;color:#555;'>"
    "Tugas Akhir — Prediksi Arah Harga BTC | VMD + Random Forest (Split 70:30)"
    "</div>",
    unsafe_allow_html=True
)
