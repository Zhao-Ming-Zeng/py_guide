import streamlit as st
import json
import os
import base64
import time
import folium
import threading
import paho.mqtt.client as mqtt
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
st.set_page_config(page_title="虎科大 IoT 智慧導覽", layout="wide", page_icon="🏫")

# --- 2. CSS 樣式 ---
st.markdown("""
<style>
    /* 隱藏預設的 audio 元素 */
    audio { display: none; }
    
    /* 美化播放按鈕 */
    .stButton button {
        background-color: #0055A4; /* 虎科藍 */
        color: white; border-radius: 50px;
        font-size: 18px; border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
        width: 100%; padding: 10px;
    }
    .stButton button:hover { background-color: #003366; transform: scale(1.02); }
    
    /* 讓地圖容器更好看 */
    iframe { border-radius: 12px; border: 2px solid #eee; }
</style>
""", unsafe_allow_html=True)

# --- 3. 載入資料 ---
if not os.path.exists("data/spots.json"):
    st.error("❌ 找不到 data/spots.json")
    st.stop()
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))

# 固定參數 (未修改)
TRIGGER_DIST = 150 
MOVE_THRESHOLD = 10 

# --- 4. 初始化 Session State ---
if 'user_coords' not in st.session_state:
    st.session_state.user_coords = None
if 'current_spot' not in st.session_state:
    st.session_state.current_spot = None
if 'mqtt_action' not in st.session_state:
    st.session_state.mqtt_action = None

# ==========================================================
# 📡 MQTT 設定 (已修正為 V2 API 以消除警告)
# ==========================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883           # Python 端必須用 1883 (TCP)
MQTT_TOPIC = "nfu/tour/control"

@st.cache_resource
def start_mqtt_listener():
    """啟動背景 MQTT 監聽"""
    
    # V2 API 的 on_connect 必須包含 properties 參數
    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"📡 MQTT 連線成功 (Code: {rc})")
        client.subscribe(MQTT_TOPIC)

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode()
            print(f"📥 收到指令: {payload}")
            # 寫入檔案作為跨執行緒溝通
            with open("mqtt_inbox.txt", "w", encoding="utf-8") as f:
                f.write(payload)
        except Exception as e:
            print(f"MQTT 錯誤: {e}")

    # 明確指定使用 VERSION2，解決 DeprecationWarning
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except:
        pass
    return client

# 啟動 MQTT
start_mqtt_listener()

# --- 5. RAG 模型 ---
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"): return "MISSING_INDEX"
    if "GOOGLE_API_KEY" not in st.secrets: return "MISSING_KEY"
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # 指定使用 gemini-2.5-flash
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3, 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
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
# 🌟 後臺監聽器 (GPS + MQTT + 防閃爍)
# ==========================================================
@st.fragment(run_every=3)
def background_worker():
    # --- A. 檢查 MQTT ---
    mqtt_cmd = None
    if os.path.exists("mqtt_inbox.txt"):
        try:
            with open("mqtt_inbox.txt", "r", encoding="utf-8") as f:
                mqtt_cmd = f.read().strip()
            os.remove("mqtt_inbox.txt")
        except: pass

    # --- B. 檢查 GPS ---
    gps_id = f"gps_{int(time.time())}"
    try:
        loc = get_geolocation(component_key=gps_id)
    except:
        loc = None
    
    # 狀態顯示 (除錯用)
    status_msg = []
    if loc: status_msg.append("🟢 GPS")
    else: status_msg.append("🔴 GPS")
    
    if mqtt_cmd:
        status_msg.append(f"⚡ IoT: {mqtt_cmd}")
        st.toast(f"收到指令: {mqtt_cmd}", icon="📡")
    
    st.caption(" | ".join(status_msg))

    # --- C. 判斷是否更新主畫面 ---
    should_rerun = False

    # 1. IoT 指令優先
    if mqtt_cmd:
        st.session_state.mqtt_action = mqtt_cmd
        should_rerun = True

    # 2. GPS 移動門檻
    if loc:
        new_lat = loc["coords"]["latitude"]
        new_lon = loc["coords"]["longitude"]
        new_pos = (new_lat, new_lon)
        
        old_pos = st.session_state.user_coords
        
        if old_pos is None:
            st.session_state.user_coords = new_pos
            should_rerun = True
        else:
            dist = geodesic(old_pos, new_pos).meters
            if dist > MOVE_THRESHOLD:
                st.session_state.user_coords = new_pos
                should_rerun = True
        
    if should_rerun:
        st.rerun()

# ==========================================================
# 主介面
# ==========================================================
st.title("虎科大隨身語音導覽")

# Sidebar
with st.sidebar:
    st.header("系統狀態")
    background_worker() # 啟動背景工人
    st.info("說明：為了節省流量並穩定畫面，只有當您移動超過 10 公尺時，地圖才會更新。")
    st.markdown(f"MQTT Topic: `{MQTT_TOPIC}`")
    st.caption("Web Client Port: 8000")

# --- 處理 MQTT 動作 ---
if st.session_state.mqtt_action:
    cmd = st.session_state.mqtt_action
    
    if cmd == "sos":
        st.error("🚨 【緊急廣播】 校園安全演練，請依照指示行動！")
        play_audio_hidden("data/audio/alert.mp3")
    elif cmd == "welcome":
        st.balloons()
        st.success("👋 歡迎蒞臨國立虎尾科技大學！")
    
    st.session_state.mqtt_action = None

col_map, col_info = st.columns([3, 2])

# --- 地圖區 ---
with col_map:
    # 虎科大預設座標 (您指定的數值)
    default_nfu_pos = (23.7027602462213, 120.42951632350216)
    
    if st.session_state.user_coords:
        center_pos = st.session_state.user_coords
        zoom = 17
    else:
        center_pos = default_nfu_pos
        zoom = 15

    m = folium.Map(location=center_pos, zoom_start=zoom)
    
    if st.session_state.user_coords:
        folium.Marker(st.session_state.user_coords, popup="我", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    
    nearest_key = None
    min_dist = float("inf")
    
    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = 99999
        if st.session_state.user_coords:
            d = geodesic(st.session_state.user_coords, spot_pos).meters
        
        folium.Marker(spot_pos, popup=f"{info['name']} ({int(d)}m)", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        
        if d < min_dist:
            min_dist = d
            nearest_key = key

    st_folium(m, width="100%", height=400)

# --- 資訊區 ---
with col_info:
    if st.session_state.user_coords and nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        st.session_state.current_spot = nearest_key
        
        st.success(f"📍 您已抵達：{spot['name']}")
        
        lang = st.radio("導覽語言", ["中文", "台語"], horizontal=True)
        intro = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "無資料")
        
        st.markdown(f"<div style='background:#f9f9f9; padding:15px; border-radius:10px; margin-bottom:10px; color:#333'>{intro}</div>", unsafe_allow_html=True)
        
        if st.button("▶ 播放導覽語音"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if suffix == "tw" and not os.path.exists(path):
                path = f"data/audio/{nearest_key}_cn.mp3"
            play_audio_hidden(path)
            
        st.divider()
        
        st.markdown("### 🤖 虎科小幫手")
        user_q = st.chat_input("有什麼問題嗎？")
        
        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(qa_chain_or_error)
            else:
                with st.chat_message("user"): st.write(user_q)
                with st.chat_message("assistant"):
                    with st.spinner("Gemini 2.5 Flash 思考中..."):
                        full_q = f"我現在在「{spot['name']}」，{user_q}"
                        resp = qa_chain_or_error.invoke(full_q)
                        st.write(resp)
                        
    elif st.session_state.user_coords:
        if nearest_key:
            st.info(f"🚶 前往最近景點：{SPOTS[nearest_key]['name']} (還有 {int(min_dist - TRIGGER_DIST)}m)")
        else:
            st.info("附近沒有景點")
    else:
        st.warning("📡 正在等待 GPS 訊號...")
        st.markdown("請確認您已開啟手機 GPS，並允许瀏覽器存取位置。")