import json
import os
import toml
import asyncio
import base64
import requests
import edge_tts

# ==============================
# 1️⃣ 讀取 API KEY
# ==============================
try:
    secrets = toml.load(".streamlit/secrets.toml")
    YATING_KEY = secrets["YATING_API_KEY"]
except Exception:
    print("❌ 無法讀取 YATING_API_KEY")
    exit(1)

# ==============================
# 2️⃣ 中文語音（Edge TTS）
# ==============================
async def gen_cn_mp3(text, path):
    print("   🎙️ [中文] 生成中...")
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice="zh-TW-HsiaoChenNeural"
        )
        await communicate.save(path)
        print("      ✅ 中文完成")
    except Exception:
        print("      ❌ 中文生成失敗")

# ==============================
# 3️⃣ 台語語音（雅婷 TTS v2）
# ==============================
def gen_tw_mp3(text, path):
    print("   🎙️ [台語] 生成中...")

    url = "https://tts.api.yating.tw/v2/speeches/short"

    headers = {
        "Content-Type": "application/json",
        "Key": YATING_KEY
    }

    payload = {
        "input": {
            "text": text,
            "type": "text"
        },
        "voice": {
            "model": "tai_female_1",
            "speed": 1.0,
            "pitch": 1.0,
            "energy": 1.0
        },
        "audioConfig": {
            "encoding": "MP3",
            "sampleRate": "16K"
        }
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=20)

        # ❌ HTTP 錯誤（不印 body，避免亂碼）
        if res.status_code not in (200, 201):
            print(f"      ❌ HTTP 錯誤：{res.status_code}")
            return
        # ❌ 非 JSON
        try:
            data = res.json()
        except Exception:
            print("      ❌ 回傳格式錯誤（非 JSON）")
            return

        audio_b64 = data.get("audioContent")
        if not audio_b64:
            print("      ❌ 回傳缺少 audioContent")
            return

        audio_bytes = base64.b64decode(audio_b64)

        with open(path, "wb") as f:
            f.write(audio_bytes)

        print("      ✅ 台語完成")      

    except requests.exceptions.Timeout:
        print("      ❌ 連線逾時")
    except requests.exceptions.RequestException:
        print("      ❌ API 連線錯誤")
    except Exception:
        print("      ❌ 未知錯誤")

# ==============================
# 4️⃣ 主程式
# ==============================
async def main():
    json_path = "data/spots.json"

    if not os.path.exists(json_path):
        print("❌ 找不到 data/spots.json")
        return

    os.makedirs("data/audio", exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("🚀 開始生成語音檔...")

    for key, info in data.items():
        print(f"\n📍 {info['name']}")

        # 中文
        cn_path = f"data/audio/{key}_cn.mp3"
        if not os.path.exists(cn_path):
            await gen_cn_mp3(info["intro_cn"], cn_path)
        else:
            print("   ℹ️ 中文檔已存在")

        # 台語
        tw_text = info.get("intro_tw", info["intro_cn"])
        tw_path = f"data/audio/{key}_tw.mp3"

        # 刪除 0kb 壞檔
        if os.path.exists(tw_path) and os.path.getsize(tw_path) < 100:
            os.remove(tw_path)

        if not os.path.exists(tw_path):
            gen_tw_mp3(tw_text, tw_path)
        else:
            print("   ℹ️ 台語檔已存在")

    print("\n🎉 全部完成")

# ==============================
# 5️⃣ 程式入口
# ==============================
if __name__ == "__main__":
    asyncio.run(main())
