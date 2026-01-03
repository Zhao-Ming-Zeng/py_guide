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

# --------------------------------------------------
# 1. 頁面設定
# --------------------------------------------------
st.set_page_config(page_title="虎科大 IoT 語音導覽", layout="wide")

# --------------------------------------------------
# 2. CSS 樣式 (極簡風格)
# --------------------------------------------------
st.markdown("""
<style>
audio { display: none; }
.stButton button {
    background-color: #0055A4; color: white; border-radius: 6px;
    font-size: 16px; border: none; width: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.stButton button:hover { background-color: #004080; }
iframe { border-radius: 8px; border: 1px solid #ddd; }
.element-container:has(.stMarkdown) { margin-bottom: 0px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. 載入資料
# --------------------------------------------------
if not os.path.exists("data/spots.json"):
    st.error("找不到 data/spots.json")
    st.stop()
SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))

TRIGGER_DIST = 150 
MOVE_THRESHOLD = 10 

# --------------------------------------------------
# 4. Session State 初始化
# --------------------------------------------------
if 'user_coords' not in st.session_state: st.session_state.user_coords = None
if 'current_spot' not in st.session_state: st.session_state.current_spot = None
if 'mqtt_action' not in st.session_state: st.session_state.mqtt_action = None
if 'last_mqtt_time' not in st.session_state: st.session_state.last_mqtt_time = 0.0

# ==========================================================
# 5. MQTT 雲端共享核心 (解決多人接收 + 雲端環境)
# ==========================================================
class MqttSharedState:
    def __init__(self):
        self.last_cmd = None
        self.timestamp = 0.0

@st.cache_resource
def get_shared_state():
    return MqttSharedState()

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "nfu/tour/control"

@st.cache_resource
def start_mqtt_listener():
    shared_state = get_shared_state()
    
    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"MQTT Connected: {rc}")
        client.subscribe(MQTT_TOPIC)

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode()
            # 更新共享記憶體
            shared_state.last_cmd = payload
            shared_state.timestamp = time.time()
        except: pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except: pass
    return client

start_mqtt_listener()

# --------------------------------------------------
# 6. RAG (AI)
# --------------------------------------------------
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"): return "請上傳 faiss_index"
    if "GOOGLE_API_KEY" not in st.secrets: return "請設定 Secrets"
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=st.secrets["GOOGLE_API_KEY"])
        prompt = PromptTemplate.from_template("背景:{context}\n問題:{question}\n回答:")
        return ({"context": db.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    except Exception as e: return str(e)

qa_chain_or_error = load_rag()

# --------------------------------------------------
# 7. 隱形播放器
# --------------------------------------------------
def play_audio_hidden(path):
    if not os.path.exists(path): return
    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    html = f"""<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
    st.markdown(html, unsafe_allow_html=True)

# ==========================================================
# 8. 背景監聽與 GPS 核心 (通用相容版)
# ==========================================================
# run_every=5 保持流暢
@st.fragment(run_every=5)
def background_worker():
    # --- A. 檢查 MQTT ---
    shared = get_shared_state()
    new_cmd = None
    try:
        if shared.timestamp > st.session_state.last_mqtt_time:
            new_cmd = shared.last_cmd
            st.session_state.last_mqtt_time = shared.timestamp
    except: pass

    # --- B. 檢查 GPS (關鍵修正：通用設定) ---
    loc = None
    try:
        # enable_high_accuracy=False -> 讓電腦可用 WiFi，手機可用快速定位
        # maximum_age=10000 -> 允許使用 10 秒內的舊位置 (防閃爍、防卡死)
        # timeout=5000 -> 5秒沒抓到就跳過
        loc = get_geolocation(
            component_key=f"gps_{int(time.time())}",
            enable_high_accuracy=False, 
            timeout=5000,
            maximum_age=10000
        )
    except:
        loc = None

    # 狀態顯示 (極簡)
    status = []
    if loc: status.append("▶ GPS 連線中") 
    else: status.append("定位中...")
    
    if new_cmd: 
        status.append(f"接收: {new_cmd}")
        st.toast(f"廣播: {new_cmd}")
        
    st.caption(" | ".join(status))

    # --- C. 刷新邏輯 ---
    should_rerun = False
    
    # 1. MQTT 觸發
    if new_cmd:
        st.session_state.mqtt_action = new_cmd
        should_rerun = True

    # 2. GPS 移動觸發
    if loc:
        try:
            new_pos = (loc["coords"]["latitude"], loc["coords"]["longitude"])
            old_pos = st.session_state.user_coords
            
            if old_pos is None:
                st.session_state.user_coords = new_pos
                should_rerun = True
            else:
                if geodesic(old_pos, new_pos).meters > MOVE_THRESHOLD:
                    st.session_state.user_coords = new_pos
                    should_rerun = True
        except: pass
                
    if should_rerun:
        st.rerun()

# ==========================================================
# 9. 主畫面 UI
# ==========================================================
st.title("虎科大 IoT 語音導覽")

with st.sidebar:
    st.header("系統狀態")
    # 自動執行背景工作
    background_worker()
    st.divider()
    st.caption(f"Topic: {MQTT_TOPIC}")

# --- MQTT 動作 ---
if st.session_state.mqtt_action:
    cmd = st.session_state.mqtt_action
    ph = st.empty()
    
    if cmd == "sos":
        ph.error("【緊急廣播】 校園安全演練，請依照指示行動！")
        play_audio_hidden("data/audio/alert.mp3")
        time.sleep(10)
    elif cmd == "welcome":
        st.balloons()
        ph.success("歡迎蒞臨國立虎尾科技大學！")
        time.sleep(5)
        
    st.session_state.mqtt_action = None
    ph.empty()
    st.rerun()

col_map, col_info = st.columns([3, 2])

# --- 地圖 ---
with col_map:
    # 預設位置
    default_pos = (23.7027602462213, 120.42951632350216)
    center = st.session_state.user_coords if st.session_state.user_coords else default_pos
    zoom = 17 if st.session_state.user_coords else 15
    
    m = folium.Map(location=center, zoom_start=zoom)
    if st.session_state.user_coords:
        folium.Marker(st.session_state.user_coords, popup="我", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    
    nearest_key = None
    min_dist = float("inf")
    
    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])
        d = geodesic(st.session_state.user_coords, spot_pos).meters if st.session_state.user_coords else 99999
        folium.Marker(spot_pos, popup=f"{info['name']} ({int(d)}m)", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        folium.Circle(spot_pos, radius=TRIGGER_DIST, color="red", fill=True, fill_opacity=0.1).add_to(m)
        if d < min_dist:
            min_dist = d
            nearest_key = key
            
    st_folium(m, width="100%", height=400)

# --- 資訊面板 ---
with col_info:
    if st.session_state.user_coords and nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        st.session_state.current_spot = nearest_key
        st.success(f"▶ 抵達：{spot['name']}")
        
        lang = st.radio("語言", ["中文", "台語"], horizontal=True)
        intro = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "無資料")
        st.markdown(f"<div style='background:#f9f9f9; padding:15px; border-radius:10px; margin-bottom:10px'>{intro}</div>", unsafe_allow_html=True)
        
        if st.button("▶ 播放語音"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if suffix == "tw" and not os.path.exists(path): path = f"data/audio/{nearest_key}_cn.mp3"
            play_audio_hidden(path)
            
        st.divider()
        user_q = st.chat_input("有問題問虎科小幫手...")
        if user_q:
            if isinstance(qa_chain_or_error, str): st.error(qa_chain_or_error)
            else:
                with st.chat_message("user"): st.write(user_q)
                with st.chat_message("assistant"):
                    with st.spinner("AI 思考中..."):
                        resp = qa_chain_or_error.invoke(f"我現在在「{spot['name']}」，{user_q}")
                        st.write(resp)
    elif st.session_state.user_coords:
        if nearest_key: st.info(f"距離 {SPOTS[nearest_key]['name']} 還有 {int(min_dist - TRIGGER_DIST)} 公尺")
    else:
        st.warning("正在定位中...")
        st.caption("請允許瀏覽器位置權限")