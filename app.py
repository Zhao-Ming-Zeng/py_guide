import streamlit as st
import json
import os
import base64
import time
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from geopy.distance import geodesic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. 設定頁面 ---
st.set_page_config(page_title="語音導覽", layout="wide", page_icon="🗺️")

# --- 2. CSS 樣式 (隱藏播放器、美化介面) ---
st.markdown("""
<style>
    /* 隱藏預設的 audio 元素 */
    audio { display: none; }
    
    /* 美化播放按鈕 */
    .stButton button {
        background-color: #E63946; color: white; border-radius: 50px;
        font-size: 18px; border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
        width: 100%; padding: 10px;
    }
    .stButton button:hover { background-color: #D62828; }
    
    /* 讓地圖容器更好看 */
    iframe { border-radius: 10px; border: 2px solid #eee; }
</style>
""", unsafe_allow_html=True)

# --- 3. 載入資料 ---
if not os.path.exists("data/spots.json"):
    st.error("❌ 找不到 data/spots.json")
    st.stop()
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))
TRIGGER_DIST = 150 # 觸發半徑
MOVE_THRESHOLD = 10 # ⚠️ 移動超過 10 公尺才更新地圖 (防閃爍核心)

# --- 4. 初始化 Session State ---
if 'user_coords' not in st.session_state:
    st.session_state.user_coords = None # 存經緯度
if 'current_spot' not in st.session_state:
    st.session_state.current_spot = None # 存目前景點

# --- 5. RAG 模型 ---
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"): return "MISSING_INDEX"
    if "GOOGLE_API_KEY" not in st.secrets: return "MISSING_KEY"
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # 設定為 2.5 Flash (或您的可用模型)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=st.secrets["GOOGLE_API_KEY"])
        prompt = PromptTemplate.from_template("背景:{context}\n問題:{question}\n回答:")
        return ({"context": db.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    except Exception as e: return str(e)

qa_chain_or_error = load_rag()

# --- 6. 隱形播放器 ---
def play_audio_hidden(path):
    if not os.path.exists(path): return
    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    # 注入一段隱形的 HTML Audio 自動播放
    html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================================
# 🌟 後臺 GPS 監聽器 (核心技術)
# ==========================================================
# 這個 fragment 會在背景每 3 秒跑一次，但「不會」刷新主頁面
@st.fragment(run_every=3)
def background_gps_worker():
    # 1. 用時間戳當 ID，強制瀏覽器抓新位置
    gps_id = f"gps_{int(time.time())}"
    
    try:
        # 這裡只會在這個小區塊顯示一個隱形的 div，不影響主畫面
        loc = get_geolocation(component_key=gps_id)
    except:
        loc = None
    
    # 顯示一個極小的狀態點，讓你知道程式還活著 (可選)
    if loc:
        st.caption(f"🟢 訊號接收中... ({int(time.time())%100})")
    else:
        st.caption("🔴 搜尋訊號中...")

    # 2. 判斷是否需要更新主畫面
    if loc:
        new_lat = loc["coords"]["latitude"]
        new_lon = loc["coords"]["longitude"]
        new_pos = (new_lat, new_lon)
        
        old_pos = st.session_state.user_coords
        
        should_update = False
        
        if old_pos is None:
            # 第一次抓到，一定要更新
            should_update = True
        else:
            # 計算移動距離
            dist = geodesic(old_pos, new_pos).meters
            # ⚠️ 只有移動距離大於門檻值 (例如 10公尺)，才觸發更新
            if dist > MOVE_THRESHOLD:
                should_update = True
        
        if should_update:
            st.session_state.user_coords = new_pos
            # 只有在這裡，才強制主畫面刷新。
            # 如果你站著不動，這行永遠不會執行，地圖就永遠不會閃！
            st.rerun()

# ==========================================================
# 主介面 (Main UI)
# ==========================================================
st.title("🗺️ 雲科大隨身語音導覽")

# 1. 啟動後臺 GPS 工人 (放在側邊欄或頁面頂端，不佔空間)
with st.sidebar:
    st.header("系統狀態")
    background_gps_worker()
    st.info("💡 說明：為了節省流量並穩定畫面，只有當您移動超過 10 公尺時，地圖才會更新。")

# 2. 處理位置與地圖
col_map, col_info = st.columns([3, 2])

with col_map:
    # 決定地圖中心
    if st.session_state.user_coords:
        center_pos = st.session_state.user_coords
        zoom = 17
    else:
        center_pos = (23.694, 120.534) # 預設雲科大
        zoom = 15

    m = folium.Map(location=center_pos, zoom_start=zoom)
    
    # 畫自己
    if st.session_state.user_coords:
        folium.Marker(st.session_state.user_coords, popup="我", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    
    # 畫景點
    nearest_key = None
    min_dist = float("inf")
    
    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        
        # 計算距離
        d = 99999
        if st.session_state.user_coords:
            d = geodesic(st.session_state.user_coords, spot_pos).meters
        
        # 標記
        folium.Marker(spot_pos, popup=f"{info['name']} ({int(d)}m)", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        
        if d < min_dist:
            min_dist = d
            nearest_key = key

    st_folium(m, width="100%", height=400)

# 3. 處理資訊面板 (這裡完全靜止，除非上面觸發 rerun)
with col_info:
    # 判斷是否抵達
    if st.session_state.user_coords and nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        
        # 更新目前景點狀態
        st.session_state.current_spot = nearest_key
        
        st.success(f"📍 您已抵達：{spot['name']}")
        
        lang = st.radio("導覽語言", ["中文", "台語"], horizontal=True)
        intro = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "無資料")
        
        st.markdown(f"<div style='background:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:10px'>{intro}</div>", unsafe_allow_html=True)
        
        # 播放按鈕
        if st.button("▶️ 播放導覽語音"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if suffix == "tw" and not os.path.exists(path):
                path = f"data/audio/{nearest_key}_cn.mp3"
            play_audio_hidden(path)
            
        st.divider()
        
        # AI 聊天
        st.markdown("### 🤖 導覽小幫手")
        user_q = st.chat_input("有什麼問題嗎？")
        
        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(qa_chain_or_error)
            else:
                with st.chat_message("user"):
                    st.write(user_q)
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        full_q = f"我現在在「{spot['name']}」，{user_q}"
                        resp = qa_chain_or_error.invoke(full_q)
                        st.write(resp)
                        
    elif st.session_state.user_coords:
        if nearest_key:
            st.info(f"🚶 前往最近景點：{SPOTS[nearest_key]['name']} (還有 {int(min_dist - TRIGGER_DIST)}m)")
            st.metric("剩餘距離", f"{int(min_dist - TRIGGER_DIST)} 公尺")
        else:
            st.info("附近沒有景點")
    else:
        st.warning("📡 正在等待 GPS 訊號...")
        st.markdown("請確認您已開啟手機 GPS，並允许瀏覽器存取位置。")