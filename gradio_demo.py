import base64
import io
from pathlib import Path
import os
from openai import OpenAI
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import gradio as gr
from easy_asr_server import EasyASR
from dotenv import load_dotenv
from typing import Union
from io import BytesIO
import queue
import asyncio
import threading
import time
import copy
import uuid
import json
import gzip
import websockets

load_dotenv()

asr = EasyASR(pipeline="sensevoice")

BASE_URL = os.getenv("DOUBAO_BASE_URL")
API_KEY = os.getenv("DOUBAO_API_KEY")
MODEL_ID = os.getenv("DOUBAO_MODEL_ID")


llm_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

system_prompt = """你是一个情感智能助手，请使用口语化、简短的回答与 user 自然地交谈。

确保你输出的所有内容都是可以被 TTS 合成为语音，不包含任何特殊字符和语音信息标记。
"""

messages = [{"role": "system", "content": system_prompt}]

class DoubaoTTS:
    def __init__(self):
        self.appid = os.getenv("DOUBAO_TTS_APPID")
        self.token = os.getenv("DOUBAO_TTS_ACCESS_TOKEN")
        self.cluster = os.getenv("DOUBAO_TTS_CLUSTER")
        self.voice_type = os.getenv("DOUBAO_TTS_VOICE_ID")
        self.host = "openspeech.bytedance.com"
        self.api_url = f"wss://{self.host}/api/v1/tts/ws_binary"
        self.sample_rate = 16000
        
        self.default_header = bytearray(b'\x11\x10\x11\x00')
        
        self.request_json = {
            "app": {
                "appid": self.appid,
                "token": self.token,
                "cluster": self.cluster
            },
            "user": {
                "uid": "388808087185088"
            },
            "audio": {
                "voice_type": self.voice_type,
                "encoding": "mp3",
                "rate": self.sample_rate,
                "speed_ratio": 1.1,
            },
            "request": {
                "reqid": "",
                "text": "",
                "text_type": "plain",
                "operation": "submit"
            }
        }

    def tts_stream(self, text: str, lang: str = "zh"):
        audio_queue = queue.Queue()
        def run_async_tts():
            asyncio.run(self._async_tts_stream(text, audio_queue))
        tts_thread = threading.Thread(target=run_async_tts)
        tts_thread.start()
        while True:
            try:
                audio_chunk = audio_queue.get(timeout=10)
                if audio_chunk is None:
                    break
                yield audio_chunk
            except queue.Empty:
                break
            time.sleep(0.01)  
        tts_thread.join()

    async def _async_tts_stream(self, text: str, audio_queue: queue.Queue):
        submit_request_json = copy.deepcopy(self.request_json)
        submit_request_json["request"]["reqid"] = str(uuid.uuid4())
        submit_request_json["request"]["text"] = text
        
        payload_bytes = str.encode(json.dumps(submit_request_json))
        payload_bytes = gzip.compress(payload_bytes)
        
        full_client_request = bytearray(self.default_header)
        full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))
        full_client_request.extend(payload_bytes)
        
        header = {"Authorization": f"Bearer; {self.token}"}
        
        try:
            async with websockets.connect(self.api_url, extra_headers=header, ping_interval=None) as ws:
                await ws.send(full_client_request)
                
                mp3_data = b""
                
                while True:
                    res = await ws.recv()
                    audio_data, done = self._parse_response(res)
                    
                    if audio_data is not None:
                        mp3_data += audio_data
                    
                    if done:
                        if mp3_data:
                            try:
                                mp3_buffer = BytesIO(mp3_data)
                                
                                audio_queue.put(mp3_buffer)
                                        
                            except Exception as e:
                                print(f"TTS 音频转换失败: {e}")
                        break
                    time.sleep(0.01)
                        
        except Exception as e:
            print(f"TTS 连接错误: {e}")
        finally:
            audio_queue.put(None)

    def _parse_response(self, res):
        message_type = res[1] >> 4
        message_type_specific_flags = res[1] & 0x0f
        header_size = res[0] & 0x0f
        payload = res[header_size*4:]
        
        if message_type == 0xb:
            if message_type_specific_flags == 0:
                return None, False
            else:
                sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                audio_data = payload[8:]
                
                if sequence_number < 0:
                    return audio_data, True
                else:
                    return audio_data, False
                    
        elif message_type == 0xf:
            code = int.from_bytes(payload[:4], "big", signed=False)
            msg_size = int.from_bytes(payload[4:8], "big", signed=False)
            error_msg = payload[8:]
            print(f"TTS错误: {code}, {error_msg}")
            return None, True
            
        return None, False

    def reset(self):
        pass

tts = DoubaoTTS()

def wav2np(input_data: Union[str, Path, BytesIO, np.ndarray], original_sr: int = 16000, target_sr: int = 16000, mono: bool = True, dtype: str = 'int16') -> np.ndarray:
    if isinstance(input_data, np.ndarray):
        audio = input_data
        original_sr = original_sr
    else:
        audio, original_sr = sf.read(input_data, dtype='float32')
    
    if mono and len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

    if dtype == 'int16':
        return np.int16(audio * 32767)
    elif dtype == 'float32':
        return np.float32(audio)
    else:
        return np.float64(audio)



def _split_av(src: str | Path):
    src = Path(src)
    stem = src.with_suffix("")

    video_path = stem.with_name(stem.name + "_video.mp4")
    audio_path = stem.with_name(stem.name + "_audio.wav")

    (
        ffmpeg
        .input(str(src))
        .output(str(video_path), c="copy", an=None)
        .overwrite_output()
        .run(quiet=True)
    )

    (
        ffmpeg
        .input(str(src))
        .output(str(audio_path), acodec="pcm_s16le", ar=48000, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )

    return video_path, audio_path


def _video_to_data_uri(video_fp: Path) -> str:
    with open(video_fp, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def _wav_to_np(wav_fp: Path, target_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(wav_fp, always_2d=False)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def prepare_llm_payload(video_obj):
    try:
        if isinstance(video_obj, dict):
            src_path = Path(video_obj["data"])
        else:
            src_path = Path(video_obj)
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    video_only, audio_only = _split_av(src_path)

    video_data_uri = _video_to_data_uri(video_only)
    audio_np       = _wav_to_np(audio_only, target_sr=16_000)

    text = asr.recognize(audio_np)

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": video_data_uri,
                    }
                },
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
    )

    response = llm_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages[-5:],
    )

    response_text = response.choices[0].message.content
    messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )
    audio_chunks = []
    for audio_np in tts.tts_stream(response_text, "zh"):
        audio_chunks.append(wav2np(audio_np, target_sr=16000, dtype='int16'))
    audio_np = np.concatenate(audio_chunks, axis=0)

    return (16000, audio_np), text, response_text



with gr.Blocks(title="视频对话") as demo:
    gr.Markdown("点击最上方 Video 组件进行录制，录制结束后自动生成回答，点击 Video 右上角 x 可清除视频进行重新录制")

    video_in   = gr.Video(label="录制视频", webcam_constraints={"video": {"width": 800, "height": 600}}, sources=["webcam", "upload"], format="mp4")
    audio_view = gr.Audio(label="回复音频", type="numpy", autoplay=True)
    text_out   = gr.Textbox(label="ASR 识别结果", max_lines=6)
    response_out = gr.Textbox(label="回复文本", max_lines=6)

    video_in.change(
        fn=prepare_llm_payload,
        inputs=video_in,
        outputs=[audio_view, text_out, response_out],
        queue=False
    )

demo.launch()