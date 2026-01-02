import streamlit as st
import json
import os
import base64
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from geopy.distance import geodesic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import requests

# =========================
# 1️⃣ 設定頁面
# =========================
st.set_page_config(page_title="語音導覽", layout="wide", page_icon="🗺️")

# =========================
# 2️⃣ CSS 美化按鈕
# =========================
st.markdown("""
<style>
.stButton button {
    background-color: #E63946; color: white; border-radius: 50%;
    width: 80px; height: 80px; font-size: 30px; border: 4px solid white;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.3); margin: 0 auto; display: block;
}
.stButton button:hover { background-color: #D62828; transform: scale(1.05); }
div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button {
    width: auto; height: auto; border-radius: 5px; font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 3️⃣ 載入景點資料
# =========================
json_path = "data/spots.json"
if not os.path.exists(json_path):
    st.error(f"❌ 找不到 {json_path}，請先執行 1a/1b 步驟！")
    st.stop()
else:
    with open(json_path, "r", encoding="utf-8") as f:
        SPOTS = json.load(f)

TRIGGER_DIST = 150  # 公尺

# =========================
# 4️⃣ RAG 模型載入
# =========================
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"):
        return "MISSING_INDEX"
    if "GOOGLE_API_KEY" not in st.secrets:
        return "MISSING_KEY"
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3, 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        prompt = PromptTemplate.from_template(
            "你是一位熱情的在地導覽員。請依據以下的背景資訊來回答遊客的問題。\n"
            "若背景資訊中沒有答案，請誠實說不知道，不要瞎掰。\n"
            "背景資訊：{context}\n"
            "遊客問題：{question}"
        )
        chain = (
            {"context": db.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()}
            | prompt 
            | llm 
            | StrOutputParser()
        )
        return chain
    except Exception as e:
        return f"ERROR: {str(e)}"

qa_chain_or_error = load_rag()

# =========================
# 5️⃣ 播放器函式
# =========================
def get_player(path):
    if not os.path.exists(path): 
        return None
    with open(path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio autoplay controls src="data:audio/mp3;base64,{b64}" style="width:100%;"></audio>'

# =========================
# 6️⃣ 主畫面
# =========================
st.title("🗺️ 雲科大隨身語音導覽")

# GPS 按鈕 & session
if 'gps_key' not in st.session_state:
    st.session_state.gps_key = 0

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 更新定位", help="點擊強制重新抓取 GPS"):
        st.session_state.gps_key += 1
        st.rerun()

# ✅ 正確 GPS 呼叫
loc = get_geolocation()

if loc:
    user_lat = loc["coords"]["latitude"]
    user_lon = loc["coords"]["longitude"]
    user_pos = (user_lat, user_lon)
    
    # 建立地圖
    m = folium.Map(location=user_pos, zoom_start=17)
    folium.Marker(user_pos, popup="您的位置", icon=folium.Icon(color="blue", icon="user")).add_to(m)

    nearest_key = None
    min_dist = float("inf")

    # 標記景點 & 計算距離
    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(user_pos, spot_pos).meters
        folium.Marker(
            spot_pos, 
            popup=f"{info['name']} ({int(d)}m)", 
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        if d < min_dist:
            min_dist = d
            nearest_key = key

    # 渲染地圖
    with col1:
        st_folium(m, width=700, height=350)

    # 觸發互動
    if nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        st.success(f"📍 您已抵達：**{spot['name']}** (距離 {int(min_dist)} 公尺)")
        
        lang = st.radio("請選擇語音導覽語言：", ["中文", "台語"], horizontal=True)
        intro_text = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "（暫無台語文字資料）")
        st.info(intro_text)

        if st.button("▶ 播放語音導覽"):
            suffix = "cn" if lang == "中文" else "tw"
            audio_path = f"data/audio/{nearest_key}_{suffix}.mp3"
            player_html = get_player(audio_path)
            if player_html:
                st.markdown(player_html, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ 音檔尚未生成：{audio_path}")

        # AI 問答
        st.divider()
        st.markdown(f"### 💬 關於 {spot['name']} 的 AI 問答")
        user_q = st.chat_input("例如：這裡有什麼歷史故事？開放時間是幾點？")
        if user_q:
            if isinstance(qa_chain_or_error, str):
                if qa_chain_or_error == "MISSING_INDEX":
                    st.error("⚠️ 找不到索引檔案。請先執行 `python 2_build_index.py`。")
                elif qa_chain_or_error == "MISSING_KEY":
                    st.error("⚠️ 找不到 Google API Key。")
                else:
                    st.error(f"⚠️ 系統錯誤：{qa_chain_or_error}")
            else:
                with st.chat_message("user"):
                    st.write(user_q)
                with st.chat_message("assistant"):
                    with st.spinner("AI 導覽員正在思考中..."):
                        full_question = f"我現在在「{spot['name']}」，請問：{user_q}"
                        response = qa_chain_or_error.invoke(full_question)
                        st.write(response)
    else:
        if nearest_key:
            st.info(f"🚶 請繼續移動... 最近景點是 **{SPOTS[nearest_key]['name']}** (還差 {int(min_dist - TRIGGER_DIST)} 公尺)")
        else:
            st.info("附近沒有已建檔的景點。")

else:
    st.warning("📡 正在取得 GPS 定位... 請允許瀏覽器存取位置權限。")
    st.write("若長時間無反應，請點擊上方的「🔄 更新定位」按鈕。")
