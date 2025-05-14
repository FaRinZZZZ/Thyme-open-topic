import streamlit as st
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor

MASTER_SERVICE = "http://127.0.0.1:8000"
MASTER_SERVICE2 = "http://127.0.0.1:8004"


# ---- CALL ENDPOINTS ----
def get_stock_analysis(ticker):
    res = requests.get(f"{MASTER_SERVICE}/ticker?ticker={ticker}")
    data = res.json()

    raw_analysis = data.get("analysis", {})
    if isinstance(raw_analysis, dict):
        data["analysis"] = raw_analysis.get("content", "")
    elif isinstance(raw_analysis, str):
        data["analysis"] = raw_analysis
    else:
        data["analysis"] = ""

    # Escape Markdown-sensitive characters for Streamlit
    data["analysis"] = data["analysis"].replace('$', '\\$')

    return data

def get_price_prediction(ticker, today):
    res = requests.post(f"{MASTER_SERVICE2}/predict", json={"ticker": ticker, "today": today})
    return res.json()

# ---- UI ----
st.set_page_config(page_title="Stock Analysis and Sentiment", page_icon="📈", layout="centered")

st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput > div > div > input {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    .main .block-container {
        max-width: 800px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Stock Analysis and Sentiment")
st.write("Enter a stock ticker symbol to get analysis and prediction.")

ticker = st.text_input("Enter Stock Ticker", "", max_chars=10)
today = datetime.today().strftime("%Y-%m-%d")

if st.button("Get Analysis and Prediction"):
    if ticker:
        with st.spinner("Fetching analysis and prediction..."):
            with ThreadPoolExecutor() as executor:
                future_analysis = executor.submit(get_stock_analysis, ticker)
                future_predict = executor.submit(get_price_prediction, ticker, today)

                analysis_result = future_analysis.result()
                prediction_result = future_predict.result()

                # st.write("🧪 analysis_result:", analysis_result)
                # st.write("🧪 analysis_result['analysis']:", analysis_result.get("analysis"))

        # # --- ANALYSIS ---
        # if "error" in analysis_result or not analysis_result.get("analysis"):
        #     st.error("Failed to fetch analysis.")
        # else:
            st.subheader("📘 LLM Analysis")
            st.markdown(analysis_result["analysis"])

        # --- PREDICTION ---
        if "error" in prediction_result:
            st.error(f"Prediction Error: {prediction_result['error']}")
        else:
            st.subheader("🔮 Predicted Stock Price (Next Day)")
            st.success(f"${prediction_result['predicted_price']:.2f}")

            st.subheader("📈 Price Trend Prediction Graph")
            img_data = base64.b64decode(prediction_result["plot_base64"])
            st.image(Image.open(BytesIO(img_data)), use_column_width=True)

    else:
        st.error("Please enter a stock ticker.")