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
from langchain_huggingface import HuggingFaceEmbeddings # 更新引用
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. 設定頁面 ---
st.set_page_config(page_title="語音導覽", layout="wide", page_icon="🗺️")

# --- 2. CSS 樣式 (美化按鈕) ---
st.markdown("""
<style>
    .stButton button {
        background-color: #E63946; color: white; border-radius: 50%;
        width: 80px; height: 80px; font-size: 30px; border: 4px solid white;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3); margin: 0 auto; display: block;
    }
    .stButton button:hover { background-color: #D62828; transform: scale(1.05); }
    /* 讓 GPS 更新按鈕正常顯示 */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stButton"] > button {
        width: auto; height: auto; border-radius: 5px; font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 載入景點資料 ---
json_path = "data/spots.json"
if not os.path.exists(json_path):
    st.error(f"❌ 找不到 {json_path}，請確認是否已執行翻譯與語音生成！")
    st.stop()
else:
    with open(json_path, "r", encoding="utf-8") as f:
        SPOTS = json.load(f)

# 設定觸發距離 (公尺)
TRIGGER_DIST = 150

# --- 4. RAG 模型載入 (修正核心錯誤) ---
@st.cache_resource
def load_rag():
    # 修正錯誤 1: 確保讀取正確的資料夾名稱 'faiss_index'
    index_path = "faiss_index"
    if not os.path.exists(index_path):
        return "MISSING_INDEX"
    
    if "GOOGLE_API_KEY" not in st.secrets:
        return "MISSING_KEY"

    try:
        # 修正錯誤 2: 強制使用 CPU，解決 NotImplementedError
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} 
        )
        
        # 載入向量資料庫
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        
        # 載入 LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3, 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        prompt = PromptTemplate.from_template(
            "你是一位熱情的在地導覽員。請依據以下的背景資訊來回答遊客的問題。\n"
            "若背景資訊中沒有答案，請誠實說不知道。\n"
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

# --- 5. 播放器函式 ---
def get_player(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<audio autoplay controls src="data:audio/mp3;base64,{b64}" style="width:100%;"></audio>'

# ================== 主畫面邏輯 ==================
st.title("🗺️ 雲科大隨身語音導覽")

# --- 6. GPS 定位邏輯 (解決不更新問題) ---
if 'gps_key' not in st.session_state:
    st.session_state.gps_key = 0

col1, col2 = st.columns([3, 1])
with col2:
    # 這裡加入 key 確保按鈕獨立
    if st.button("🔄 更新定位", key="btn_refresh"):
        st.session_state.gps_key += 1
        st.rerun()

# 關鍵：每次 gps_key 改變，get_geolocation 就會被強制視為新的元件重新執行
gps_component_key = "gps_" + str(st.session_state.gps_key)
loc = get_geolocation(key=gps_component_key)

if loc:
    user_lat = loc["coords"]["latitude"]
    user_lon = loc["coords"]["longitude"]
    user_pos = (user_lat, user_lon)
    
    # --- 7. 顯示地圖 ---
    m = folium.Map(location=user_pos, zoom_start=17)
    folium.Marker(user_pos, popup="您的位置", icon=folium.Icon(color="blue", icon="user")).add_to(m)
    
    nearest_key = None
    min_dist = float("inf")

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

    with col1:
        st_folium(m, width=700, height=350)
    
    # --- 8. 互動區 (語言按鈕回來了) ---
    if nearest_key and min_dist <= TRIGGER_DIST:
        spot = SPOTS[nearest_key]
        st.success(f"📍 您已抵達：**{spot['name']}** (距離 {int(min_dist)} 公尺)")
        
        # 語言選擇 (原本不見的功能加回來)
        lang = st.radio("請選擇語音導覽語言：", ["中文", "台語"], horizontal=True)
        
        intro_text = spot["intro_cn"] if lang == "中文" else spot.get("intro_tw", "（暫無台語文字資料）")
        st.info(intro_text)
        
        if st.button("▶ 播放語音導覽"):
            suffix = "cn" if lang == "中文" else "tw"
            # 這裡特別處理：如果台語檔不存在，嘗試用中文檔頂替 (避免報錯)
            audio_path = f"data/audio/{nearest_key}_{suffix}.mp3"
            if not os.path.exists(audio_path) and suffix == "tw":
                audio_path = f"data/audio/{nearest_key}_cn.mp3"
            
            player_html = get_player(audio_path)
            if player_html:
                st.markdown(player_html, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ 音檔尚未生成")

        # --- 9. AI 問答區 ---
        st.divider()
        st.markdown(f"### 💬 關於 {spot['name']} 的 AI 問答")
        
        user_q = st.chat_input("例如：這裡有什麼歷史故事？")
        
        if user_q:
            if isinstance(qa_chain_or_error, str):
                if qa_chain_or_error == "MISSING_INDEX":
                    st.error("⚠️ 錯誤：找不到 faiss_index 資料夾。請確認您執行了 2_build_index.py")
                elif qa_chain_or_error == "MISSING_KEY":
                    st.error("⚠️ 錯誤：secrets.toml 缺少 Google API Key。")
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
        st.info(f"🚶 請繼續移動... 最近的景點是 **{SPOTS[nearest_key]['name']}** (還差 {int(min_dist - TRIGGER_DIST)} 公尺)")

else:
    st.warning("📡 正在取得 GPS 定位... 若無反應請點擊「更新定位」。")