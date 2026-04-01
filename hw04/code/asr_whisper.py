"""
开源ASR实现：基于OpenAI Whisper的语音识别程序
用于识别剪映导出的voice_clone.mp3音频
"""

import whisper
import os
from pathlib import Path

# 1. 配置路径（自动适配你的文件夹结构，不用改）
ROOT_DIR = Path(__file__).parent.parent
AUDIO_PATH = ROOT_DIR / "assets" / "voice_clone.mp3"
OUTPUT_PATH = ROOT_DIR / "assets" / "asr_result.txt"

# 2. 加载Whisper模型（base模型，体积小，适合笔记本，不用GPU也能跑）
print("正在加载Whisper base模型...")
model = whisper.load_model("base")

# 3. 检查音频文件是否存在
if not AUDIO_PATH.exists():
    raise FileNotFoundError(f"音频文件不存在！请检查路径：{AUDIO_PATH}\n请确保voice_clone.mp3已放入hw04/assets/文件夹")

# 4. 执行语音识别
print(f"正在识别音频：{AUDIO_PATH}")
result = model.transcribe(str(AUDIO_PATH), language="zh")  # 指定中文，识别更准确

# 5. 保存识别结果
print("识别完成，正在保存结果...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("=== Whisper ASR 识别结果 ===\n")
    f.write(f"音频文件：{AUDIO_PATH.name}\n")
    f.write(f"使用模型：whisper-base\n")
    f.write("\n识别文本：\n")
    f.write(result["text"])
    f.write("\n\n分段时间戳：\n")
    for segment in result["segments"]:
        f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}\n")

print(f"识别结果已保存到：{OUTPUT_PATH}")
print("\n识别文本预览：")
print(result["text"])
