import sounddevice as sd
from collections import deque
import numpy as np
from ten_vad import TenVad
import time
import threading
import litecam
import requests
import json
import base64
import os
from PIL import Image
from io import BytesIO
from openai import OpenAI

from easy_tts_server import create_tts_engine

from easy_asr_server import EasyASR

from dotenv import load_dotenv
load_dotenv()

asr = EasyASR(pipeline="sensevoice")

block_size = 256

bot_speaking = False

in_queue = deque()

vad = TenVad(block_size, 0.5)

human_voice_queue = deque()

def audio_io_callback(indata, frames, time, status):
    global bot_speaking
    indata = indata.astype(np.int16)
    if bot_speaking:
        in_queue.append((indata, False))
    else:
        prob, human_speaking = vad.process(indata.squeeze())
        in_queue.append((indata, human_speaking))

audio_io_stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype='int16',
    blocksize=block_size,
    callback=audio_io_callback,
)

audio_io_stream.start()

time.sleep(2)
in_queue.clear()


def get_human_speech_thread():
    silence_cnt = 0
    human_voice_buffer = []

    while True:
        if len(in_queue):
            indata, human_speaking = in_queue.popleft()
            if human_speaking:
                silence_cnt = 0
                human_voice_buffer.append(indata)
            else:
                silence_cnt = min(silence_cnt + 1, 20)
                if silence_cnt <= 10:
                    human_voice_buffer.append(indata)
                else:
                    if len(human_voice_buffer) > 8000 // block_size:
                        voice_segment = np.concatenate(human_voice_buffer, axis=0)
                        human_voice_queue.append(voice_segment)
                        print(f"Human speaking: {voice_segment.shape}")
                    human_voice_buffer.clear()
        time.sleep(0.01)

threading.Thread(target=get_human_speech_thread, daemon=True).start()

human_speech_queue = deque()


def asr_thread():
    while True:
        if len(human_voice_queue):
            voice_segment = human_voice_queue.popleft()
            text = asr.recognize(voice_segment.squeeze().astype(np.float32))
            human_speech_queue.append(text)
            print(f"Human speech: {text}")
        time.sleep(0.01)

threading.Thread(target=asr_thread, daemon=True).start()

camera = litecam.PyCamera()
camera_queue = deque(maxlen=1)

def camera_thread():
    if camera.open(0):
        # window = litecam.PyWindow(
        #     camera.getWidth() // 4, camera.getHeight() // 4, "Camera Stream")

        while True:
            frame = camera.captureFrame()
            if frame is not None:
                width = frame[0]
                height = frame[1]
                size = frame[2]
                data = frame[3]
                # window.showFrame(width, height, data)
                
                # Convert RGB data to base64 for OpenRouter
                try:
                    # Convert raw RGB data to PIL Image
                    rgb_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                    image = Image.fromarray(rgb_array, 'RGB')
                    
                    # Convert to JPEG format in memory
                    buffer = BytesIO()
                    image.save(buffer, format='JPEG', quality=85)
                    buffer.seek(0)
                    
                    # Encode to base64
                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    data_url = f"data:image/jpeg;base64,{base64_image}"
                    
                    # Add to camera queue
                    camera_queue.append(data_url)
                    
                except Exception as e:
                    print(f"Error processing camera frame: {e}")
                    
            time.sleep(1)  # ~30 FPS

    camera.release()

llm_response_queue = deque()


def llm_thread():


    API_KEY = os.getenv("DOUBAO_API_KEY")
    if not API_KEY:
        print("Warning: LLM_API_KEY not set")
        return
    
    BASE_URL = os.getenv("DOUBAO_BASE_URL")
    if not BASE_URL:
        print("Warning: DOUBAO_BASE_URL not set")
        return

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    messages = [{"role": "system","content": """你是一个情感丰富，善解人意的心理咨询师。你擅长通过语言和视觉信息来理解用户的心理状态，并给出合理的回答。你回答简洁且口语化。"""},]
    
    while True:
        if len(human_speech_queue) > 0 and len(camera_queue) > 0:
            try:
                # Get text input and image
                text_input = ""
                image_data_url = ""
                while len(human_speech_queue) > 0:
                    text_input += human_speech_queue.popleft()
                if len(camera_queue) > 0:
                    image_data_url = camera_queue.popleft()
                
                # Create messages for OpenRouter
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text_input
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                )
                
                response = client.chat.completions.create(
                    model="doubao-1.5-vision-lite-250315",
                    messages=messages,
                    stream=True,
                )
                
                llm_response = ""

                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        llm_response += chunk.choices[0].delta.content

                llm_response_queue.append(llm_response)
                print(f"LLM Response: {llm_response}")
                    
            except Exception as e:
                print(f"Error in LLM thread: {e}")
                
        time.sleep(0.1)

threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=llm_thread, daemon=True).start()

tts = create_tts_engine()

def tts_thread():
    global bot_speaking
    
    while True:
        if len(llm_response_queue):
            text = llm_response_queue.popleft()
            print(f"TTS echo: {text}")
            # audio_np = tts.tts(text, "zh")
            # bot_speaking = True
            # sd.play(audio_np, tts.sample_rate)
            # sd.wait()
            bot_speaking = True
            for audio_np in tts.tts_stream(text, "zh"):
                sd.play(audio_np, tts.sample_rate)
                sd.wait()
            bot_speaking = False
        time.sleep(0.01)


threading.Thread(target=tts_thread, daemon=True).start()


print("启动完成")

while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        break

camera.release()
