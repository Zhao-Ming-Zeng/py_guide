import json
import os
import toml
import asyncio
import requests
import base64
import edge_tts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 載入金鑰
try:
    secrets = toml.load(".streamlit/secrets.toml")
    GOOGLE_KEY = secrets["GOOGLE_API_KEY"]
    YATING_KEY = secrets["YATING_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY
except:
    print("❌ 請檢查 .streamlit/secrets.toml")
    exit()

# 建立輸出目錄
os.makedirs("data/audio", exist_ok=True)

# 1. 翻譯函式 (Gemini)
def translate_to_tw(text):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    tpl = """請將中文改寫為「台灣閩南語（台語）的口語漢字」，不要用羅馬拼音，直接輸出結果。
    原文：{text}"""
    return llm.invoke(tpl).content.strip()

# 2. 中文語音 (Edge TTS)
async def gen_cn_mp3(text, filename):
    print(f"   🎙️ 生成中文語音: {filename}...")
    communicate = edge_tts.Communicate(text, "zh-TW-HsiaoChenNeural")
    await communicate.save(filename)

# 3. 台語語音 (雅婷 API)
def gen_tw_mp3(text, filename):
    print(f"   🎙️ 生成台語語音 (雅婷): {filename}...")
    url = "https://api.yating.tw/v2/text-to-speech/synthesize"
    headers = {"Authorization": f"Key {YATING_KEY}", "Content-Type": "application/json"}
    payload = {
        "input": {"text": text, "type": "text"},
        "voice": {"model": "zh_en_female_1", "speed": 1.0, "pitch": 1.0},
        "audioConfig": {"encoding": "MP3", "sampleRate": 22050}
    }
    try:
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 201:
            audio_content = base64.b64decode(res.json()["audioContent"])
            with open(filename, "wb") as f:
                f.write(audio_content)
        else:
            print(f"   ❌ 雅婷 API 錯誤: {res.text}")
    except Exception as e:
        print(f"   ❌ 連線錯誤: {e}")

async def main():
    with open('data/spots.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("🚀 開始批次生成資源...")

    for key, info in data.items():
        print(f"\n📍 處理景點：{info['name']} ({key})")
        
        # A. 翻譯
        if not info.get('intro_tw'):
            print("   🔄 翻譯台語文稿中...")
            info['intro_tw'] = translate_to_tw(info['intro_cn'])
        
        # B. 生成中文 MP3
        cn_path = f"data/audio/{key}_cn.mp3"
        if not os.path.exists(cn_path):
            await gen_cn_mp3(info['intro_cn'], cn_path)
        
        # C. 生成台語 MP3
        tw_path = f"data/audio/{key}_tw.mp3"
        if not os.path.exists(tw_path):
            gen_tw_mp3(info['intro_tw'], tw_path)

    # 更新 JSON (把翻譯好的台語存回去)
    with open('data/spots.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 所有資源生成完畢！請檢查 data/audio/ 資料夾。")

if __name__ == "__main__":
    asyncio.run(main())