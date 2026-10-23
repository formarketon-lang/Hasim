
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from io import BytesIO
from datetime import datetime

st.set_page_config(
    page_title="TradeLab Pro Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- THEME --------------------
st.markdown("""
<style>
.stApp {background:#0b1120;color:#e2e8f0}
[data-testid="stSidebar"] {background:#111827}
div[data-testid="stMetric"] {
    background:#1e293b;
    border:1px solid #334155;
    padding:14px;border-radius:12px
}
.stButton>button {
    background:#0891b2;color:white;
    border-radius:8px;min-height:42px;
    border:0;font-weight:600
}
h1,h2,h3 {color:#38bdf8}
</style>
""", unsafe_allow_html=True)

st.title("📈 TradeLab Pro")
st.caption("STOCK SCANNER | NSE • BSE • CUSTOM DATA")

# -------------------- CONSTANTS --------------------
INDEX_BASKETS = {
    "Broad Market": [
        "Nifty 50", "Nifty Next 50", "Nifty 100",
        "Nifty 200", "Nifty 500", "Nifty Total Market"
    ],
    "Market Cap Indices": [
        "Nifty LargeMidcap 250", "Nifty Midcap 50",
        "Nifty Midcap 100", "Nifty Midcap 150",
        "Nifty Smallcap 50", "Nifty Smallcap 100",
        "Nifty Smallcap 250", "Nifty Microcap 250"
    ],
    "Sectoral Indices": [
        "Nifty Auto", "Nifty Bank", "Nifty Financial Services",
        "Nifty FMCG", "Nifty Healthcare", "Nifty IT",
        "Nifty Media", "Nifty Metal", "Nifty Pharma",
        "Nifty Realty", "Nifty Consumer Durables",
        "Nifty Oil & Gas", "Nifty PSU Bank",
        "Nifty Private Bank", "Nifty Energy",
        "Nifty Infrastructure", "Nifty Chemicals",
        "Nifty Capital Markets", "Nifty Defence"
    ],
    "Thematic Indices": [
        "Nifty India Consumption", "Nifty India Manufacturing",
        "Nifty PSE", "Nifty CPSE", "Nifty Commodities",
        "Nifty MNC", "Nifty Services Sector",
        "Nifty Mobility", "Nifty Housing"
    ],
    "BSE Indices": [
        "BSE Sensex", "BSE 100", "BSE 150 MidCap",
        "BSE 250 SmallCap", "BSE 500", "BSE MidCap",
        "BSE SmallCap", "BSE Bankex", "BSE Healthcare",
        "BSE IT", "BSE Auto", "BSE FMCG",
        "BSE Capital Goods", "BSE Metal", "BSE Realty",
        "BSE Oil & Gas"
    ]
}

# -------------------- DEMO DATA --------------------
@st.cache_data
def make_demo_data():
    symbols = [
        ("RELIANCE", "Energy", "Large"),
        ("HDFCBANK", "Banking", "Large"),
        ("ICICIBANK", "Banking", "Large"),
        ("SBIN", "Banking", "Large"),
        ("INFY", "IT", "Large"),
        ("TCS", "IT", "Large"),
        ("HINDUNILVR", "FMCG", "Large"),
        ("ITC", "FMCG", "Large"),
        ("BHARTIARTL", "Telecom", "Large"),
        ("LT", "Capital Goods", "Large"),
        ("MARUTI", "Automobile", "Large"),
        ("TATAMOTORS", "Automobile", "Large"),
        ("SUNPHARMA", "Pharma", "Large"),
        ("TATASTEEL", "Metals", "Large"),
        ("BEL", "Defence", "Large"),
        ("TRENT", "Retail", "Large"),
        ("ZOMATO", "Technology", "Large"),
        ("DIXON", "Electronics", "Mid"),
        ("POLYCAB", "Electricals", "Mid"),
        ("PERSISTENT", "IT", "Mid"),
        ("COFORGE", "IT", "Mid"),
        ("FEDERALBNK", "Banking", "Mid"),
        ("BSE", "Financial Services", "Mid"),
        ("MCX", "Financial Services", "Mid"),
        ("ASHOKLEY", "Automobile", "Mid"),
        ("VOLTAS", "Consumer Durables", "Mid"),
        ("CROMPTON", "Consumer Durables", "Mid"),
        ("AETHER", "Chemicals", "Small"),
        ("DATAPATTNS", "Defence", "Small"),
        ("KAYNES", "Electronics", "Small"),
    ]

    rng = np.random.default_rng(42)
    rows = []

    for symbol, sector, cap in symbols:
        close = round(float(rng.uniform(100, 3000)), 2)
        prev = round(close * (1 + float(rng.uniform(-0.04, 0.04))), 2)
        opn = round(prev * (1 + float(rng.uniform(-0.01, 0.01))), 2)
        high = round(max(opn, close) * 1.015, 2)
        low = round(min(opn, close) * 0.985, 2)
        volume = int(rng.integers(100000, 5000000))

        rows.append({
            "Symbol": symbol,
            "Exchange": "NSE",
            "Index": "Nifty 500",
            "Sector": sector,
            "MarketCap": cap,
            "Open": opn,
            "High": high,
            "Low": low,
            "Close": close,
            "PrevClose": prev,
            "Volume": volume,
        })

    df = pd.DataFrame(rows)
    df["ChangePct"] = (
        (df["Close"] - df["PrevClose"])
        / df["PrevClose"] * 100
    ).round(2)

    return df

# -------------------- DATA NORMALIZATION --------------------
COLUMN_ALIASES = {
    "symbol": "Symbol",
    "ticker": "Symbol",
    "stock": "Symbol",
    "exchange": "Exchange",
    "index": "Index",
    "index_name": "Index",
    "sector": "Sector",
    "marketcap": "MarketCap",
    "market_cap": "MarketCap",
    "market cap": "MarketCap",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Close",
    "adj_close": "Close",
    "prevclose": "PrevClose",
    "previous_close": "PrevClose",
    "volume": "Volume",
}

REQUIRED_COLUMNS = [
    "Symbol", "Exchange", "Open", "High",
    "Low", "Close", "Volume"
]

def normalize_data(df):
    if df is None or df.empty:
        raise ValueError("ডেটা খালি।")

    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace("-", "_")
        for c in df.columns
    ]
    df = df.rename(columns={
        c: COLUMN_ALIASES[c]
        for c in df.columns if c in COLUMN_ALIASES
    })

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "প্রয়োজনীয় কলাম পাওয়া যায়নি: "
            + ", ".join(missing)
        )

    if "Index" not in df.columns:
        df["Index"] = "Unknown"
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"
    if "MarketCap" not in df.columns:
        df["MarketCap"] = "Unknown"
    if "PrevClose" not in df.columns:
        df["PrevClose"] = np.nan

    for col in [
        "Open", "High", "Low", "Close",
        "PrevClose", "Volume"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df["Exchange"] = df["Exchange"].astype(str).str.upper().str.strip()
    df["MarketCap"] = df["MarketCap"].astype(str).str.title().str.strip()

    df = df.dropna(subset=["Symbol", "Close", "Open", "High", "Low"])
    df = df[df["Close"] > 0]

    df["ChangePct"] = np.where(
        df["PrevClose"].notna() & (df["PrevClose"] > 0),
        (df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100,
        np.nan
    )
    return df

# -------------------- FOUR UNIVERSE OPTIONS --------------------
st.subheader("Universe Scanning")

with st.container(border=True):
    st.markdown("### 1 · Data Source")

    source = st.selectbox(
        "ডেটা কোথা থেকে নেবেন?",
        [
            "Demo Data",
            "CSV Import",
            "Excel Import",
            "Manual Data Entry",
            "Custom API"
        ]
    )

    raw_df = None

    if source == "Demo Data":
        raw_df = make_demo_data()
        st.info("Demo data শুধু অ্যাপ পরীক্ষা করার জন্য; এটি লাইভ মার্কেট ডেটা নয়।")

    elif source == "CSV Import":
        uploaded = st.file_uploader(
            "CSV ফাইল নির্বাচন করুন",
            type=["csv"],
            key="csv_upload"
        )
        if uploaded:
            try:
                raw_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"CSV পড়া যায়নি: {e}")

    elif source == "Excel Import":
        uploaded = st.file_uploader(
            "Excel ফাইল নির্বাচন করুন",
            type=["xlsx", "xls"],
            key="excel_upload"
        )
        if uploaded:
            try:
                raw_df = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"Excel পড়া যায়নি: {e}")

    elif source == "Manual Data Entry":
        st.caption("একটি নমুনা সারি দেখানো হয়েছে। নিজের ডেটা পরিবর্তন বা নতুন সারি যোগ করুন।")
        template = pd.DataFrame([{
            "Symbol": "RELIANCE",
            "Exchange": "NSE",
            "Index": "Nifty 500",
            "Sector": "Energy",
            "MarketCap": "Large",
            "Open": 1400.0,
            "High": 1420.0,
            "Low": 1390.0,
            "Close": 1410.0,
            "PrevClose": 1400.0,
            "Volume": 1000000
        }])
        edited = st.data_editor(
            template,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_editor"
        )
        if st.button("Manual Data গ্রহণ করুন"):
            try:
                st.session_state["manual_data"] = normalize_data(edited)
                st.success("Manual data গ্রহণ করা হয়েছে।")
            except Exception as e:
                st.error(str(e))

        raw_df = st.session_state.get("manual_data")

    elif source == "Custom API":
        st.caption(
            "GET endpoint থেকে JSON অথবা CSV ডেটা আনুন। "
            "API প্রদানকারীর অনুমতি ও ব্যবহারের শর্ত মেনে চলুন।"
        )
        api_url = st.text_input(
            "API Endpoint URL",
            placeholder="https://example.com/api/stocks"
        )
        api_token = st.text_input(
            "API Key / Bearer Token (ঐচ্ছিক)",
            type="password"
        )
        headers_text = st.text_area(
            "অতিরিক্ত Headers — JSON",
            value="{}",
            help='উদাহরণ: {"Accept": "application/json"}'
        )
        params_text = st.text_area(
            "Query Parameters — JSON",
            value="{}",
            help='উদাহরণ: {"symbol": "RELIANCE"}'
        )
        response_format = st.selectbox(
            "Response Format",
            ["Auto Detect", "JSON", "CSV"]
        )
        timeout = st.slider(
            "Timeout (seconds)", 5, 60, 20
        )

        if st.button("Connect API & Fetch Data"):
            if not api_url.strip():
                st.error("API URL দিন।")
            else:
                try:
                    headers = json.loads(headers_text or "{}")
                    params = json.loads(params_text or "{}")

                    if not isinstance(headers, dict) or not isinstance(params, dict):
                        raise ValueError("Headers ও Parameters JSON object হতে হবে।")

                    if api_token:
                        headers["Authorization"] = (
                            api_token if api_token.lower().startswith("bearer ")
                            else "Bearer " + api_token
                        )

                    response = requests.get(
                        api_url.strip(),
                        headers=headers,
                        params=params,
                        timeout=timeout
                    )
                    response.raise_for_status()

                    fmt = response_format
                    content_type = response.headers.get("Content-Type", "").lower()

                    if fmt == "CSV" or (
                        fmt == "Auto Detect" and
                        ("csv" in content_type or
                         api_url.lower().split("?")[0].endswith(".csv"))
                    ):
                        api_df = pd.read_csv(BytesIO(response.content))
                    else:
                        payload = response.json() if fmt != "CSV" else None

                        # Common API wrappers
                        if isinstance(payload, dict):
                            for key in ("data", "results", "stocks", "items"):
                                if key in payload and isinstance(payload[key], list):
                                    payload = payload[key]
                                    break
                        if not isinstance(payload, list):
                            raise ValueError(
                                "API response-এ একটি JSON list বা CSV table প্রয়োজন।"
                            )
                        api_df = pd.DataFrame(payload)

                    normalized = normalize_data(api_df)
                    st.session_state["api_data"] = normalized
                    st.success(f"API থেকে {len(normalized)}টি রো এসেছে।")

                except Exception as e:
                    st.error(f"API fetch ব্যর্থ: {e}")

        raw_df = st.session_state.get("api_data")

    # Normalize imported data
    if raw_df is not None:
        try:
            df = normalize_data(raw_df)
            st.session_state["active_data"] = df
        except Exception as e:
            st.error(f"ডেটা যাচাই ব্যর্থ: {e}")
            df = None
    else:
        df = st.session_state.get("active_data")

with st.container(border=True):
    st.markdown("### 2 · Exchange Selection")
    exchange = st.selectbox("Exchange", ["NSE", "BSE", "Both"])

with st.container(border=True):
    st.markdown("### 3 · Index Basket (A–Z)")

    all_indices = sorted(
        {idx for values in INDEX_BASKETS.values() for idx in values},
        key=str.lower
    )
    selected_indices = st.multiselect(
        "Index নির্বাচন করুন",
        all_indices,
        default=["Nifty 500"]
    )
    st.caption(
        "একটি ডেটা ফাইলে Index কলাম থাকলে সেটি দিয়ে ফিল্টার হবে। "
        "কোনো Index-এর সম্পূর্ণ constituent list স্বয়ংক্রিয়ভাবে তৈরি হয় না।"
    )

with st.container(border=True):
    st.markdown("### 4 · Market Cap")
    market_caps = st.multiselect(
        "Market Cap নির্বাচন করুন",
        ["Large", "Mid", "Small", "Micro", "Unknown"],
        default=["Large", "Mid", "Small", "Unknown"]
    )

# -------------------- SCAN --------------------
st.divider()
st.subheader("Scanner Controls")

c1, c2, c3 = st.columns(3)
with c1:
    positive_only = st.checkbox("শুধু Positive Change", value=False)
with c2:
    min_volume = st.number_input(
        "Minimum Volume", min_value=0, value=0, step=10000
    )
with c3:
    min_price = st.number_input(
        "Minimum Price (₹)", min_value=0.0, value=0.0, step=10.0
    )

if st.button("🔎 RUN SCANNER", use_container_width=True):
    if df is None or df.empty:
        st.error("প্রথমে একটি Data Source নির্বাচন করে বৈধ ডেটা দিন।")
    else:
        result = df.copy()

        if exchange != "Both":
            result = result[result["Exchange"] == exchange]

        # Filter only when the dataset contains mapped Index labels.
        if selected_indices and "Index" in result.columns:
            result = result[
                result["Index"].astype(str).isin(selected_indices)
            ]

        if market_caps:
            result = result[
                result["MarketCap"].astype(str).str.title().isin(market_caps)
            ]
        else:
            result = result.iloc[0:0]

        result = result[
            (result["Close"] >= min_price) &
            (result["Volume"] >= min_volume)
        ]

        if positive_only:
            result = result[result["ChangePct"] > 0]

        result = result.sort_values(
            "ChangePct", ascending=False, na_position="last"
        )

        st.session_state["scan_result"] = result
        st.session_state["scan_time"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# -------------------- RESULTS --------------------
result = st.session_state.get("scan_result")

if result is not None:
    st.markdown("## Scan Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Stocks Found", len(result))
    m2.metric(
        "Positive",
        int((result["ChangePct"] > 0).sum())
    )
    m3.metric(
        "Negative",
        int((result["ChangePct"] < 0).sum())
    )

    st.caption(
        "Last scan: " + st.session_state.get("scan_time", "—")
    )

    display_cols = [
        "Symbol", "Exchange", "Index", "Sector", "MarketCap",
        "Open", "High", "Low", "Close", "PrevClose",
        "ChangePct", "Volume"
    ]
    display_cols = [c for c in display_cols if c in result.columns]

    st.dataframe(
        result[display_cols],
        use_container_width=True,
        hide_index=True
    )

    csv_data = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Export Scan Results CSV",
        data=csv_data,
        file_name="tradelabpro_scan_results.csv",
        mime="text/csv"
    )

    st.markdown("### Price Overview")
    if not result.empty:
        chart_data = result.set_index("Symbol")[["Close"]]
        st.bar_chart(chart_data)

st.divider()
st.caption(
    "TradeLab Pro | Educational scanner. "
    "Demo data is synthetic. This tool does not provide investment advice."
)
