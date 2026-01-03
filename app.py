import streamlit as st
import json
import os
import time
import folium
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
# CSS (優化手機版顯示)
# --------------------------------------------------
st.markdown("""
<style>
.stButton button {
    background-color: #0055A4;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    height: 3em;
    width: 100%;
}
/* 隱藏 audio播放器但保留功能 */
.stAudio {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
if not os.path.exists("data/spots.json"):
    st.error("找不到 data/spots.json，請檢查檔案路徑。")
    st.stop()

SPOTS = json.load(open("data/spots.json", "r", encoding="utf-8"))

TRIGGER_DIST = 150
MOVE_THRESHOLD = 5  # 降低移動門檻以增加靈敏度

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "user_coords" not in st.session_state:
    st.session_state.user_coords = None
if "last_coords" not in st.session_state:
    st.session_state.last_coords = None
if "current_spot" not in st.session_state:
    st.session_state.current_spot = None
if "mqtt_action" not in st.session_state:
    st.session_state.mqtt_action = None
if "last_mqtt_time" not in st.session_state:
    st.session_state.last_mqtt_time = 0.0
# 用於控制音效播放
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None

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
            # 寫入暫存檔
            with open(MQTT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"MQTT Connect Error: {e}")

    return client

start_mqtt_listener()

# --------------------------------------------------
# RAG (錯誤處理增強)
# --------------------------------------------------
@st.cache_resource
def load_rag():
    if not os.path.exists("faiss_index"):
        return "FAISS index missing"
    if "GOOGLE_API_KEY" not in st.secrets:
        return "GOOGLE_API_KEY missing in st.secrets"

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
            "背景:{context}\n問題:{question}\n回答 (請簡短，適合語音朗讀):"
        )
        return (
            {"context": db.as_retriever(search_kwargs={"k": 2}),
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    except Exception as e:
        return f"RAG Error: {str(e)}"

qa_chain_or_error = load_rag()

# --------------------------------------------------
# Logic Functions
# --------------------------------------------------
def check_mqtt():
    """檢查是否有新的廣播指令"""
    if os.path.exists(MQTT_FILE):
        try:
            with open(MQTT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content: return None
                data = json.loads(content)
            
            server_time = data.get("timestamp", 0)
            cmd = data.get("cmd", "")

            if server_time > st.session_state.last_mqtt_time:
                st.session_state.last_mqtt_time = server_time
                return cmd
        except (json.JSONDecodeError, OSError):
            pass # 忽略讀取衝突
    return None

# --------------------------------------------------
# Background worker (只負責觸發 Rerun)
# --------------------------------------------------
@st.fragment(run_every=4)  # 放慢到 4 秒，給 GPS 緩衝時間
def background_worker():
    # 這個 Fragment 的唯一作用就是定期喚醒 Streamlit
    # 讓主腳本重新執行，進而觸發 GPS 讀取和 MQTT 檢查
    
    # 檢查 MQTT (雖然主腳本也會查，但這裡可以加快反應)
    cmd = check_mqtt()
    if cmd:
        st.session_state.mqtt_action = cmd
        st.rerun()
    
    # 這裡不做 GPS 讀取，因為 GPS 讀取必須在主線程渲染元件
    # 單純的 Rerun 就會觸發下方的 get_geolocation

    # 可以在這裡印個隱形的時間戳，確保它在跑
    st.empty()

# --------------------------------------------------
# 主程式邏輯
# --------------------------------------------------

st.title("虎科大 IoT 智慧導覽")

# 1. 啟動背景計時器 (放在 Sidebar 以免影響排版)
with st.sidebar:
    st.header("系統狀態")
    background_worker()
    st.info("系統運作中...請保持螢幕開啟")
    
    # 2. 獲取 GPS (關鍵修正：放在主線程，不使用變動 Key)
    # enableHighAccuracy=True 對 Android 非常重要
    loc = get_geolocation(
        enableHighAccuracy=True, 
        timeout=10000, 
        maximumAge=0
    )

# 3. 處理 GPS 數據
if loc and "coords" in loc:
    new_lat = loc["coords"]["latitude"]
    new_lon = loc["coords"]["longitude"]
    new_pos = (new_lat, new_lon)

    # 只有當位置改變超過閾值，或這是第一次定位時，才更新
    old_pos = st.session_state.user_coords
    
    if old_pos is None:
        st.session_state.user_coords = new_pos
        # 第一次定位不 Rerun，直接往下跑
    else:
        dist = geodesic(old_pos, new_pos).meters
        if dist > MOVE_THRESHOLD:
            st.session_state.user_coords = new_pos
            # 位置大幅變動，自動 Rerun 以更新地圖
            st.rerun()

elif st.session_state.user_coords is None:
    # 還沒抓到位置時的提示
    st.warning("正在獲取精確位置 (Android 請稍候 5-10 秒)...")

# 4. 處理 MQTT 指令
cmd = check_mqtt()
if cmd:
    st.session_state.mqtt_action = cmd

if st.session_state.mqtt_action:
    action = st.session_state.mqtt_action
    if action == "sos":
        st.error("【緊急廣播】 校園安全演練！")
        st.session_state.audio_to_play = "data/audio/alert.mp3"
        time.sleep(3) # 給使用者看一眼
    elif action == "welcome":
        st.balloons()
        st.success("歡迎蒞臨國立虎尾科技大學")
    
    st.session_state.mqtt_action = None
    st.rerun()

# 5. UI 佈局
col_map, col_info = st.columns([3, 2])

with col_map:
    # 預設位置
    center_pos = st.session_state.user_coords if st.session_state.user_coords else (23.7027, 120.4295)
    zoom = 18 if st.session_state.user_coords else 15

    m = folium.Map(location=center_pos, zoom_start=zoom)

    # 畫使用者
    if st.session_state.user_coords:
        folium.Marker(
            st.session_state.user_coords,
            popup="Current Location",
            icon=folium.Icon(color="red", icon="user")
        ).add_to(m)

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
        folium.Marker(
            spot_pos,
            popup=f"{info['name']} ({int(d)}m)"
        ).add_to(m)

        # 觸發圈
        folium.Circle(
            spot_pos,
            radius=TRIGGER_DIST,
            fill=True,
            color="#3388ff",
            fill_opacity=0.1
        ).add_to(m)

        if d < min_dist:
            min_dist = d
            nearest_key = key

    st_folium(m, width="100%", height=450, key="main_map")

with col_info:
    # 6. 播放音效 (隱藏式播放器，利用 state 控制)
    if st.session_state.audio_to_play:
        try:
            st.audio(st.session_state.audio_to_play, format="audio/mp3", autoplay=True)
            # 播放後清除，避免重整頁面時重播，但要小心清除太快導致沒播出來
            # 這裡不立即清除，讓使用者下次互動或移動時才消失
            st.session_state.audio_to_play = None
        except Exception:
            pass

    # 7. 抵達判斷
    if st.session_state.user_coords and nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        
        # 如果剛抵達這個新景點，自動切換
        if st.session_state.current_spot != nearest_key:
            st.session_state.current_spot = nearest_key
            st.toast(f"已抵達：{spot['name']}")

        st.success(f"📍 您在：{spot['name']}")
        
        lang = st.radio("導覽語言", ["中文", "台語"], horizontal=True, key="lang_select")
        
        # 顯示介紹
        intro_text = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "無資料")
        st.write(intro_text)

        # 手動播放按鈕
        if st.button("▶ 播放導覽"):
            suffix = "cn" if lang == "中文" else "tw"
            path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if not os.path.exists(path) and lang == "台語":
                 path = f"data/audio/{nearest_key}_cn.mp3" # Fallback
            
            st.session_state.audio_to_play = path
            st.rerun()

        st.divider()
        st.markdown("🤖 **虎科小幫手**")
        
        user_q = st.chat_input("關於這裡的問題...")
        if user_q:
            if isinstance(qa_chain_or_error, str):
                st.error(qa_chain_or_error)
            else:
                with st.spinner("思考中..."):
                    resp = qa_chain_or_error.invoke(f"我現在在「{spot['name']}」，{user_q}")
                    st.write(resp)
                    
    elif st.session_state.user_coords:
        if nearest_key:
            st.info(f"距離最近：{SPOTS[nearest_key]['name']} (約 {int(min_dist)} 公尺)")
        else:
            st.info("附近無景點")
    else:
        st.warning("等待 GPS 定位訊號...")
        st.write("請確認：")
        st.write("1. 手機 GPS 已開啟")
        st.write("2. 瀏覽器已允許使用位置")