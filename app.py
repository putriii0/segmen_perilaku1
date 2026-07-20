import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentasi Konsumen | K-Means",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e8eaf6;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}

/* ── Section header ── */
.section-header {
    font-size: 20px;
    font-weight: 600;
    color: #1a237e;
    border-left: 4px solid #3949ab;
    padding-left: 12px;
    margin: 24px 0 12px;
}

/* ── Cluster card ── */
.cluster-card {
    border-radius: 16px;
    padding: 28px 32px;
    color: #fff;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,.18);
    margin-bottom: 20px;
}

/* ── Info box ── */
.info-box {
    background: #e8eaf6;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    border-left: 4px solid #3949ab;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #9e9e9e;
    font-size: 13px;
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("model/scaler.pkl")
        kmeans = joblib.load("model/kmeans_model.pkl")

        return scaler, kmeans

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    path = Path("data/hasil_segmentasi.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cluster_col = next(
        (c for c in df.columns if c.lower() in ["cluster", "segment"]), None
    )
    if cluster_col and cluster_col != "cluster":
        df.rename(columns={cluster_col: "cluster"}, inplace=True)
    return df


segment_name = {
    0: "Pelanggan Premium",
    1: "Pelanggan Toko Fisik",
    2: "Pelanggan Online",
    3: "Pelanggan Sesekali",
    4: "Pelanggan Aktif Hemat",
    5: "Pelanggan Bernilai Tinggi",
    6: "Pelanggan Loyal Toko",
    7: "Pelanggan Digital"
}

segment_icon = {
    "Pelanggan Premium": "💎",
    "Pelanggan Toko Fisik": "🏪",
    "Pelanggan Online": "🛒",
    "Pelanggan Sesekali": "🙂",
    "Pelanggan Aktif Hemat": "💰",
    "Pelanggan Bernilai Tinggi": "👑",
    "Pelanggan Loyal Toko": "❤️",
    "Pelanggan Digital": "📱"
}

segment_description = {

    "Pelanggan Premium":
    "Memiliki pengeluaran tinggi baik secara online maupun di toko fisik.",

    "Pelanggan Toko Fisik":
    "Lebih sering berbelanja dan melakukan pengeluaran melalui toko fisik.",

    "Pelanggan Online":
    "Lebih aktif melakukan transaksi dan pengeluaran melalui platform online.",

    "Pelanggan Sesekali":
    "Memiliki frekuensi transaksi dan pengeluaran yang relatif rendah.",

    "Pelanggan Aktif Hemat":
    "Sering melakukan transaksi, namun dengan nilai pembelian yang relatif kecil.",

    "Pelanggan Bernilai Tinggi":
    "Memiliki frekuensi transaksi dan pengeluaran yang tinggi pada seluruh kanal belanja.",

    "Pelanggan Loyal Toko":
    "Sering bertransaksi dengan pengeluaran yang lebih besar di toko fisik.",

    "Pelanggan Digital":
    "Sangat aktif melakukan pembelian secara online dengan pengeluaran online yang tinggi."
}

strategy = {

    "Pelanggan Premium":
    "Memberikan layanan eksklusif, promo khusus, dan program loyalitas premium.",

    "Pelanggan Toko Fisik":
    "Meningkatkan pengalaman belanja di toko melalui promosi, pelayanan yang lebih baik, dan program loyalitas.",

    "Pelanggan Online":
    "Mengoptimalkan promosi digital, rekomendasi produk, serta kemudahan transaksi online.",

    "Pelanggan Sesekali":
    "Memberikan promosi dan insentif agar pelanggan lebih sering melakukan pembelian.",

    "Pelanggan Aktif Hemat":
    "Menawarkan paket hemat, diskon, dan bundling untuk meningkatkan nilai transaksi.",

    "Pelanggan Bernilai Tinggi":
    "Mempertahankan loyalitas pelanggan melalui reward eksklusif, pelayanan prioritas, dan penawaran premium.",

    "Pelanggan Loyal Toko":
    "Mengembangkan program loyalitas khusus pelanggan yang aktif berbelanja di toko fisik.",

    "Pelanggan Digital":
    "Meningkatkan pengalaman belanja digital melalui personalisasi dan promosi online."
}

cluster_colors = {
    0:"#F39C12",
    1:"#8E44AD",
    2:"#3498DB",
    3:"#95A5A6",
    4:"#27AE60",
    5:"#E74C3C",
    6:"#16A085",
    7:"#2C3E50"
}



# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Segmentasi Konsumen")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["📊 Dataset & Visualisasi", "🔮 Demo Prediksi Cluster"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Model: <b>K-Means Clustering</b><br>Fitur: 4 variabel perilaku<br>Cluster: 8 segmen</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("<small style='color:#78909c'>© 2026 DASD_Kelompok_6</small>", unsafe_allow_html=True)
    


# ════════════════════════════════════════════════════════
# PAGE 1 — DATASET & VISUALISASI
# ════════════════════════════════════════════════════════
if page == "📊 Dataset & Visualisasi":
    # Header
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a237e,#3949ab);border-radius:16px;padding:32px 36px;color:#fff;margin-bottom:28px'>
        <h1 style='margin:0;font-size:2rem'>🛒 Segmentasi Perilaku Belanja Konsumen</h1>
        <p style='margin:8px 0 0;opacity:.85;font-size:1rem'>
            Analisis pola belanja menggunakan K-Means Clustering untuk mengidentifikasi
            segmen konsumen berdasarkan 4 fitur perilaku.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    if df is None:
        st.error("⚠️ File `data/hasil_segmentasi.csv` tidak ditemukan.")
        st.stop()

    # ── Metrics ─────────────────────────────────────────
    n_clusters = df["cluster"].nunique() if "cluster" in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Data", f"{len(df):,}")
    c2.metric("🔢 Jumlah Fitur", len([c for c in df.columns if c != "cluster"]))
    c3.metric("🎯 Jumlah Cluster", n_clusters)
    c4.metric("🤖 Algoritma", "K-Means")

    # ── Distribusi Cluster ───────────────────────────────
    st.markdown("<div class='section-header'>📊 Distribusi Cluster</div>", unsafe_allow_html=True)

    if "cluster" in df.columns:

        dist = df["cluster"].value_counts().sort_index().reset_index()
        dist.columns = ["Cluster", "Jumlah"]

        dist["Label"] = dist["Cluster"].map(
            lambda x: f"Cluster {x}: {segment_name[x]}"
        )

        dist["Warna"] = dist["Cluster"].map(
            lambda x: cluster_colors[x]
        )

        col_bar, col_pie = st.columns(2)

        with col_bar:
            fig_bar = px.bar(
                dist,
                x="Label",
                y="Jumlah",
                color="Label",
                color_discrete_sequence=list(cluster_colors.values()),
                title="Jumlah Konsumen per Cluster",
                text="Jumlah",
            )

            fig_bar.update_traces(
                textposition="outside",
                marker_line_width=0
            )

            fig_bar.update_layout(
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_family="Inter",
                title_font_size=15,
                xaxis=dict(title=""),
                yaxis=dict(title="Jumlah Konsumen"),
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pie:
            fig_pie = px.pie(
                dist,
                names="Label",
                values="Jumlah",
                color="Label",
                color_discrete_sequence=list(cluster_colors.values()),
                title="Proporsi Tiap Cluster",
                hole=0.45,
            )

            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Pemetaan Segmen Pelanggan ─────────────────────────

    segment_profile = (
        df.groupby("cluster")[
            [
                "monthly_online_orders",
                "monthly_store_visits",
                "avg_online_spend",
                "avg_store_spend"
            ]
        ]
        .mean()
        .reset_index()
    )

    segment_profile["Segmen"] = segment_profile["cluster"].map(segment_name)

    fig = px.bar(
        segment_profile,
        x="Segmen",
        y="avg_online_spend",
        color="Segmen",
        title="Rata-rata Pengeluaran Online per Segmen",
        text="avg_online_spend"
    )

    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")

    fig.update_layout(
        showlegend=False,
        xaxis_title="Segmen",
        yaxis_title="Rata-rata Pengeluaran Online",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)


    # ── Profil Segmen Konsumen ─────────────────────────────
    st.markdown(
        "<div class='section-header'>📋 Profil Segmen Konsumen</div>",
        unsafe_allow_html=True
    )

    profile = (
        df.groupby("cluster")[
            [
                "monthly_online_orders",
                "monthly_store_visits",
                "avg_online_spend",
                "avg_store_spend"
            ]
        ]
        .mean()
        .round(2)
    )

    profile.columns = [
        "Pesanan Online/Bulan",
        "Kunjungan Toko/Bulan",
        "Rata-rata Belanja Online",
        "Rata-rata Belanja Offline"
    ]

    st.dataframe(profile, use_container_width=True)

    for i in sorted(segment_name.keys()):

        nama = segment_name[i]

        icon = segment_icon[nama]

        deskripsi = segment_description[nama]

        st.info(f"{icon} **{nama}**\n\n{deskripsi}")



# ════════════════════════════════════════════════════════
# PAGE 2 — DEMO PREDIKSI
# ════════════════════════════════════════════════════════

# ═══════════════════════════════════════
# PAGE 2 — DEMO PREDIKSI
# ═══════════════════════════════════════
else:

    # Load model
    scaler, kmeans = load_models()

    if scaler is None:
        st.stop()

    st.markdown("### 👤 Data Konsumen")

    col1, col2 = st.columns(2)

    with col1:
        online_orders = st.number_input(
            "🛒 Jumlah Pesanan Online/Bulan",
            min_value=0,
            value=5
        )

        online_spend = st.number_input(
            "💰 Rata-rata Belanja Online (PKR)",
            min_value=523,
            max_value=149996,
            value=75000,
            step=1000,
            format="%d"
        )

    with col2:
        store_visits = st.number_input(
            "🏪 Jumlah Kunjungan Toko/Bulan",
            min_value=0,
            value=3
        )

        store_spend = st.number_input(
            "🏪 Rata-rata Belanja Offline (PKR)",
            min_value=542,
            max_value=149972,
            value=75000,
            step=1000,
            format="%d"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Prediksi Cluster", use_container_width=True):

        input_df = pd.DataFrame(
            [[
                online_orders,
                store_visits,
                online_spend,
                store_spend
            ]],
            columns=[
                "monthly_online_orders",
                "monthly_store_visits",
                "avg_online_spend",
                "avg_store_spend"
            ]
        )

        input_scaled = scaler.transform(input_df)

        cluster_pred = int(kmeans.predict(input_scaled)[0])

        nama = segment_name[cluster_pred]
        deskripsi = segment_description[nama]
        icon = segment_icon[nama]
        strategi = strategy[nama]
        warna = cluster_colors[cluster_pred]

        st.markdown(
            f"""
            <div class='cluster-card' style='background:{warna}'>
                <h2>{icon} Cluster {cluster_pred}</h2>
                <h3>{nama}</h3>
                <p>{deskripsi}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    

        # Strategy & summary
        s_col, t_col = st.columns([1, 1])

        with s_col:
            st.markdown("#### 📌 Rekomendasi Strategi Pemasaran")
            st.success(strategi)

        with t_col:
            st.markdown("#### 📋 Ringkasan Input")

            summary = pd.DataFrame({
                "Fitur": [
                    "Jumlah Pesanan Online/Bulan",
                    "Jumlah Kunjungan Toko/Bulan",
                    "Rata-rata Belanja Online",
                    "Rata-rata Belanja Offline"
                ],
                "Nilai": [
                    online_orders,
                    store_visits,
                    f"PKR {online_spend:,.0f}",
                    f"PKR {store_spend:,.0f}"
                ]
            })

            st.dataframe(summary, hide_index=True, use_container_width=True)


        # ── Visualisasi Profil Konsumen ───────────────────────
        st.markdown("### 📊 Profil Konsumen")

        profile_df = pd.DataFrame({
            "Fitur": [
                "Online Orders",
                "Store Visits",
                "Online Spend",
                "Store Spend"
            ],
            "Nilai": [
                online_orders/49,
                store_visits/19,
                online_spend/149996,
                store_spend/149972
            ]
        })

        fig = px.line_polar(
            profile_df,
            r="Nilai",
            theta="Fitur",
            line_close=True
        )

        fig.update_traces(fill="toself")

        st.plotly_chart(fig, use_container_width=True)


        st.markdown(
            "<div class='footer'>Segmentasi Perilaku Belanja Konsumen · K-Means Clustering</div>",
            unsafe_allow_html=True,
        )