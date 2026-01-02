import streamlit as st
import json
import os
import base64
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

# --- 2. 自動刷新機制 (確保 GPS 更新) ---
# 設定 3 秒刷新一次
count = st_autorefresh(interval=3000, key="gps_updater")

# --- 3. CSS 樣式 ---
st.markdown("""
<style>
    .stButton button {
        background-color: #E63946; color: white; border-radius: 50%;
        width: 80px; height: 80px; font-size: 30px; border: 4px solid white;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3); margin: 0 auto; display: block;
    }
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button {
        width: auto; height: auto; border-radius: 5px; font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 載入資料 ---
if not os.path.exists("data/spots.json"):
    st.error("❌ 找不到 data/spots.json")
    st.stop()
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))
TRIGGER_DIST = 150

# --- 5. RAG 模型 (設定為 Flash) ---
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"): return "MISSING_INDEX"
    if "GOOGLE_API_KEY" not in st.secrets: return "MISSING_KEY"

    try:
        # Embeddings (CPU 模式)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # 🌟 設定模型為 gemini-1.5-flash (目前最快的 Flash 版本)
        # 如果未來真的出了 2.5，請將字串改為 "gemini-2.5-flash"
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3, 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        prompt = PromptTemplate.from_template(
            "你是在地導覽員。依據背景回答，不知道就說不知道。\n背景:{context}\n問題:{question}"
        )
        return ({"context": db.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    except Exception as e:
        return f"ERROR: {str(e)}"

qa_chain_or_error = load_rag()

def get_player(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    return f'<audio autoplay controls src="data:audio/mp3;base64,{b64}" style="width:100%;"></audio>'

# ================== 主畫面 ==================
st.title("🗺️ 雲科大隨身語音導覽")

# --- 6. GPS 定位邏輯 ---
# 初始化位置記憶
if 'last_pos' not in st.session_state:
    st.session_state.last_pos = None

col1, col2 = st.columns([3, 1])
with col2:
    st.caption(f"📡 GPS 更新中... ({count})")
    if st.button("手動更新"): st.rerun()

# 每次刷新換 ID，強制更新
gps_id = f"gps_{count}"
try:
    loc = get_geolocation(component_key=gps_id)
except:
    loc = None

# 如果抓到位置，更新記憶
if loc:
    st.session_state.last_pos = loc

# 優先使用當下位置，否則用記憶位置
current_loc = loc if loc else st.session_state.last_pos

# --- 7. 地圖顯示邏輯 ---
if current_loc:
    lat = current_loc["coords"]["latitude"]
    lon = current_loc["coords"]["longitude"]
    user_pos = (lat, lon)
    
    # 建立地圖
    m = folium.Map(location=user_pos, zoom_start=17)
    folium.Marker(user_pos, popup="我", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    
    nearest_key = None
    min_dist = float("inf")

    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(user_pos, spot_pos).meters
        folium.Marker(spot_pos, popup=f"{info['name']} ({int(d)}m)", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        if d < min_dist:
            min_dist = d
            nearest_key = key

    with col1:
        st_folium(m, width=700, height=350)
    
    # --- 8. 互動區 ---
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
            player = get_player(path)
            if player: st.markdown(player, unsafe_allow_html=True)

        st.divider()
        user_q = st.chat_input(f"問問關於 {spot['name']} 的事...")
        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(qa_chain_or_error)
            else:
                with st.spinner("AI 思考中..."):
                    resp = qa_chain_or_error.invoke(f"地點:{spot['name']}, 問題:{user_q}")
                    st.write(resp)
    else:
        st.info(f"🚶 前往最近景點：{SPOTS[nearest_key]['name']} (還有 {int(min_dist - TRIGGER_DIST)}m)")

else:
    # ⚠️ 如果地圖沒出來，代表連第一次定位都還沒抓到
    st.warning("📡 正在衛星定位中... 請稍候")
    # 這裡顯示一個預設地圖 (雲科大)，避免畫面全白
    default_pos = (23.694, 120.534) 
    m_default = folium.Map(location=default_pos, zoom_start=15)
    with col1:
        st_folium(m_default, width=700, height=350)