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
# Page config
# --------------------------------------------------
st.set_page_config(page_title="虎科大 IoT 智慧導覽", layout="wide")

# --------------------------------------------------
# CSS
# --------------------------------------------------
st.markdown("""
<style>
audio { display: none; }
.stButton button {
    background-color: #0055A4;
    color: white;
    border-radius: 6px;
    font-size: 16px;
    border: none;
    width: 100%;
}
iframe {
    border-radius: 8px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
if not os.path.exists("data/spots.json"):
    st.error("spots.json not found")
    st.stop()

SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))

TRIGGER_DIST = 150
MOVE_THRESHOLD = 10

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "user_coords" not in st.session_state:
    st.session_state.user_coords = None

if "current_spot" not in st.session_state:
    st.session_state.current_spot = None

if "mqtt_action" not in st.session_state:
    st.session_state.mqtt_action = None

# MQTT 多人接收修正
if "last_mqtt_time" not in st.session_state:
    st.session_state.last_mqtt_time = 0.0

# --------------------------------------------------
# MQTT (JSON 廣播模式)
# --------------------------------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "nfu/tour/control"
MQTT_FILE = "mqtt_broadcast.json"

@st.cache_resource
def start_mqtt_listener():
    def on_connect(client, userdata, flags, rc, properties=None):
        client.subscribe(MQTT_TOPIC)

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode()
            data = {
                "cmd": payload,
                "timestamp": time.time()
            }
            with open(MQTT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except:
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except:
        pass

    return client

start_mqtt_listener()

# --------------------------------------------------
# RAG
# --------------------------------------------------
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"):
        return "FAISS index missing"
    if "GOOGLE_API_KEY" not in st.secrets:
        return "GOOGLE_API_KEY missing"

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )

        prompt = PromptTemplate.from_template(
            "背景:{context}\n問題:{question}\n回答:"
        )

        return (
            {"context": db.as_retriever(search_kwargs={"k": 2}),
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    except Exception as e:
        return str(e)

qa_chain_or_error = load_rag()

# --------------------------------------------------
# Audio
# --------------------------------------------------
def play_audio_hidden(path):
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(html, unsafe_allow_html=True)

# --------------------------------------------------
# Background worker
# --------------------------------------------------
@st.fragment(run_every=3) # 改回原本的 3 秒
def background_worker():

    # 1. MQTT 檢查
    mqtt_cmd = None
    if os.path.exists(MQTT_FILE):
        try:
            with open(MQTT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            server_time = data.get("timestamp", 0)
            cmd = data.get("cmd", "")

            if server_time > st.session_state.last_mqtt_time:
                mqtt_cmd = cmd
                st.session_state.last_mqtt_time = server_time
        except:
            pass

    # 2. GPS 檢查 (完全改回最原始版本，不帶任何參數)
    loc = None
    try:
        gps_id = f"gps_{int(time.time())}"
        # ⚠️ 這裡改回了最單純的呼叫，這就是您說電腦可以秒抓的狀態
        loc = get_geolocation(component_key=gps_id)
    except:
        loc = None

    should_rerun = False

    if mqtt_cmd:
        st.session_state.mqtt_action = mqtt_cmd
        should_rerun = True

    if loc:
        new_pos = (
            loc["coords"]["latitude"],
            loc["coords"]["longitude"]
        )

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

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("虎科大隨身語音導覽")

with st.sidebar:
    st.header("系統狀態")

    # 保留手動按鈕以備不時之需，但不需要先按它才能跑
    if st.button("🔄 重抓位置"):
        st.rerun()

    # 自動執行，不需啟用按鈕
    background_worker()

    st.info("系統正在自動定位中")
    st.markdown(f"MQTT Topic: {MQTT_TOPIC}")

# --------------------------------------------------
# MQTT action
# --------------------------------------------------
if st.session_state.mqtt_action:
    cmd = st.session_state.mqtt_action

    if cmd == "sos":
        st.error("【緊急廣播】 校園安全演練，請依照指示行動！")
        play_audio_hidden("data/audio/alert.mp3")
        time.sleep(10)

    elif cmd == "welcome":
        st.balloons()
        st.success("歡迎蒞臨國立虎尾科技大學")
        time.sleep(5)

    st.session_state.mqtt_action = None
    st.rerun()

# --------------------------------------------------
# Layout
# --------------------------------------------------
col_map, col_info = st.columns([3, 2])

with col_map:

    default_nfu_pos = (23.7027602462213, 120.42951632350216)

    if st.session_state.user_coords:
        center_pos = st.session_state.user_coords
        zoom = 17
    else:
        center_pos = default_nfu_pos
        zoom = 15

    m = folium.Map(location=center_pos, zoom_start=zoom)

    if st.session_state.user_coords:
        folium.Marker(
            st.session_state.user_coords,
            popup="我"
        ).add_to(m)

    nearest_key = None
    min_dist = float("inf")

    for key, info in SPOTS.items():
        spot_pos = (info["lat"], info["lon"])

        if st.session_state.user_coords:
            d = geodesic(st.session_state.user_coords, spot_pos).meters
        else:
            d = 99999

        folium.Marker(
            spot_pos,
            popup=f"{info['name']} ({int(d)}m)"
        ).add_to(m)

        folium.Circle(
            spot_pos,
            radius=TRIGGER_DIST,
            fill=True,
            fill_opacity=0.1
        ).add_to(m)

        if d < min_dist:
            min_dist = d
            nearest_key = key

    st_folium(m, width="100%", height=400)

with col_info:

    if st.session_state.user_coords and nearest_key and min_dist <= TRIGGER_DIST:

        spot = SPOTS[nearest_key]
        st.session_state.current_spot = nearest_key

        st.success(f"您已抵達：{spot['name']}")

        lang = st.radio("導覽語言", ["中文", "台語"], horizontal=True)

        intro = (
            spot["intro_cn"]
            if lang == "中文"
            else spot.get("intro_tw", "無資料")
        )

        st.markdown(intro)

        if st.button("▶ 播放導覽語音"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if suffix == "tw" and not os.path.exists(path):
                path = f"data/audio/{nearest_key}_cn.mp3"
            play_audio_hidden(path)

        st.markdown("虎科小幫手")

        user_q = st.chat_input("請輸入問題")

        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(qa_chain_or_error)
            else:
                with st.chat_message("user"):
                    st.write(user_q)
                with st.chat_message("assistant"):
                    resp = qa_chain_or_error.invoke(
                        f"我現在在「{spot['name']}」，{user_q}"
                    )
                    st.write(resp)

    elif st.session_state.user_coords:
        if nearest_key:
            st.info(
                f"前往最近景點：{SPOTS[nearest_key]['name']}，"
                f"距離 {int(min_dist - TRIGGER_DIST)} 公尺"
            )
        else:
            st.info("附近沒有景點")
    else:
        st.warning("等待 GPS 定位")