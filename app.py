import streamlit as st
import json
import os
import base64
import time
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from streamlit_autorefresh import st_autorefresh
from geopy.distance import geodesic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. 設定頁面 ---
st.set_page_config(page_title="語音導覽", layout="wide", page_icon="🗺️")

# --- 2. 自動刷新機制 (您要求的固定時間) ---
# 固定 3000 毫秒 (3秒) 刷新一次，強制更新 GPS
refresh_count = st_autorefresh(interval=3000, key="gps_updater")

# --- 3. CSS 樣式 ---
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

# --- 4. 載入資料 ---
json_path = "data/spots.json"
if not os.path.exists(json_path):
    st.error(f"❌ 找不到 {json_path}")
    st.stop()
else:
    with open(json_path, "r", encoding="utf-8") as f:
        SPOTS = json.load(f)

# 觸發半徑 (進入範圍才播放)
TRIGGER_DIST = 150

# --- 5. RAG 模型 (依照指示：保留您原始設定，不動) ---
@st.cache_resource
def load_rag():
    index_path = "faiss_index"
    if not os.path.exists(index_path): return "MISSING_INDEX"
    if "GOOGLE_API_KEY" not in st.secrets: return "MISSING_KEY"

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # 這裡保留標準設定，若您有自己的模型參數可直接在此修改
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3, 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        prompt = PromptTemplate.from_template(
            "導覽員背景知識：{context}\n遊客問題：{question}\n請依據背景回答，若無資訊請說RAG無該內容。"
        )
        
        chain = (
            {"context": db.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        return chain
    except Exception as e:
        return f"ERROR: {str(e)}"

qa_chain_or_error = load_rag()

# --- 6. 播放器 ---
def get_player(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<audio autoplay controls src="data:audio/mp3;base64,{b64}" style="width:100%;"></audio>'

# ================== 主畫面 ==================
st.title("🗺️ 雲科大隨身語音導覽")

# --- 7. GPS 定位邏輯 (已刪除最小距離限制) ---

col1, col2 = st.columns([3, 1])
with col2:
    st.caption(f"📡 GPS 更新計數: {refresh_count}")
    if st.button("手動更新"):
        st.rerun()

# 每次都用新的 Key，確保不使用 Streamlit 快取
gps_id = f"gps_{refresh_count}"

try:
    # 這裡加入 maximumAge=0 強制不使用瀏覽器位置快取
    # enableHighAccuracy=True 要求最高精準度
    current_loc = get_geolocation(
        component_key=gps_id,
        enableHighAccuracy=True,
        maximumAge=0, 
        timeout=5000
    )
except TypeError:
    # 相容舊版參數
    current_loc = get_geolocation(key=gps_id)

# 只要抓到位置就直接更新，完全不判斷距離差
loc = current_loc

if loc:
    user_lat = loc["coords"]["latitude"]
    user_lon = loc["coords"]["longitude"]
    user_pos = (user_lat, user_lon)
    
    # --- 8. 地圖顯示 ---
    m = folium.Map(location=user_pos, zoom_start=17)
    # 藍色點：您的位置
    folium.Marker(
        user_pos, 
        popup="我", 
        icon=folium.Icon(color="blue", icon="user")
    ).add_to(m)
    
    nearest_key = None
    min_dist = float("inf")

    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(user_pos, spot_pos).meters
        
        # 紅色點：景點
        folium.Marker(
            spot_pos, 
            popup=f"{info['name']} ({int(d)}m)", 
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        # 紅色圈：觸發範圍
        folium.Circle(
            spot_pos, 
            radius=TRIGGER_DIST, 
            color="red", 
            fill=True, 
            fill_opacity=0.1
        ).add_to(m)
        
        if d < min_dist:
            min_dist = d
            nearest_key = key

    with col1:
        st_folium(m, width=700, height=350)
    
    # --- 9. 觸發與互動 ---
    if nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        st.success(f"📍 抵達：**{spot['name']}**")
        
        lang = st.radio("語言", ["中文", "台語"], horizontal=True)
        intro_text = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "無資料")
        st.info(intro_text)
        
        if st.button("▶ 播放"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if suffix == "tw" and not os.path.exists(path):
                path = f"data/audio/{nearest_key}_cn.mp3"
                st.warning("⚠️ 暫無台語檔，播放中文")
            player = get_player(path)
            if player: st.markdown(player, unsafe_allow_html=True)

        st.divider()
        user_q = st.chat_input("有什麼問題想問導覽員？")
        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(f"系統錯誤: {qa_chain_or_error}")
            else:
                with st.spinner("AI 思考中..."):
                    resp = qa_chain_or_error.invoke(f"地點:{spot['name']}, 問題:{user_q}")
                    st.write(resp)
    else:
        st.info(f"🚶 前往最近景點：{SPOTS[nearest_key]['name']} (還有 {int(min_dist - TRIGGER_DIST)}m)")

else:
    st.warning("📡 正在取得 GPS 定位... (每 3 秒更新)")