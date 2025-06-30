import os
import torch
import ffmpeg
import hashlib
import folder_paths
import numpy as np
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from .uvr5.mdxnet import MDXNetDereverb
from .uvr5.vr import AudioPre, AudioPreDeEcho
import tempfile
import torchaudio
import requests
from tqdm import tqdm

input_path = folder_paths.get_input_directory()
output_path = folder_paths.get_output_directory()
base_path = os.path.dirname(__file__)
models_dir = folder_paths.models_dir
node_path = os.path.dirname(os.path.dirname(base_path))
weights_path = os.path.join(models_dir, "uvr5_weights")
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True

# 创建权重目录
os.makedirs(weights_path, exist_ok=True)

def download_file(url, destination):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"下载失败: HTTP {response.status_code} - {url}")
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def get_model_path(model_name):
    """获取模型路径，如果不存在则下载"""
    # 使用新的权重路径 weights_path
    # 如果是onnx模型
    if model_name == "onnx_dereverb_By_FoxJoy":
        model_dir = os.path.join(weights_path, "onnx_dereverb_By_FoxJoy")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "vocals.onnx")
        if not os.path.exists(model_path):
            url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx"
            download_file(url, model_path)
        return model_path
    
    # 其他模型直接保存在 weights_path 下
    model_path = os.path.join(weights_path, model_name)
    if not os.path.exists(model_path):
        # 使用正确的下载URL
        url = f"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/{model_name}"
        download_file(url, model_path)
    
    return model_path
    

def load_audio_file(file_path):
    """加载音频文件为ComfyUI兼容格式"""
    waveform, sample_rate = torchaudio.load(file_path)
    # 修改为官方兼容的3D格式 [1, channels, samples]
    return {
        "waveform": waveform.unsqueeze(0),
        "sample_rate": sample_rate
    }



class UVR5:
    @classmethod
    def INPUT_TYPES(s):
        model_list = [
            "HP5_only_main_vocal.pth", 
            "HP5-主旋律人声vocals+其他instrumentals.pth",
            "HP2_all_vocals.pth", 
            "HP2-人声vocals+非人声instrumentals.pth",
            "HP3_all_vocals.pth", 
            "VR-DeEchoAggressive.pth",
            "VR-DeEchoDeReverb.pth", 
            "VR-DeEchoNormal.pth", 
            "onnx_dereverb_By_FoxJoy"
        ]
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (model_list, {"default": "HP5_only_main_vocal.pth"}),
                "agg": ("INT", {
                    "default": 10, 
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "display": "slider"
                }),
                "formato": (["wav", "flac", "mp3", "m4a"], {"default": "wav"})
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocal_audio", "bgm_audio")

    FUNCTION = "split"

    CATEGORY = "audio"

    def split(self, audio, model, agg, formato):
        # 获取模型路径（会自动下载）
        model_path = get_model_path(model)
        
        # 处理音频
        vocal_audio, bgm_audio = self.process_uvr(audio, model, agg, formato, model_path)
        return (vocal_audio, bgm_audio)

    def process_uvr(self, audio, model_name, agg, formato, model_path):
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 保存输入音频到临时文件
            input_path = os.path.join(tmp_dir, "input.wav")
            
            # 关键修改：处理3D音频张量 [batch, channels, samples]
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # 如果音频是3D的（官方格式），转换为2D用于保存
            if waveform.dim() == 3:
                # 取第一个批次（我们只处理单批次）
                waveform_to_save = waveform[0]
            else:
                waveform_to_save = waveform
                
            torchaudio.save(input_path, waveform_to_save, sample_rate)
            
            # 创建输出路径
            save_root_vocal = os.path.join(tmp_dir, "vocal")
            save_root_ins = os.path.join(tmp_dir, "instrument")
            os.makedirs(save_root_vocal, exist_ok=True)
            os.makedirs(save_root_ins, exist_ok=True)
            
            # 处理音频
            is_hp3 = "HP3" in model_name
            if model_name == "onnx_dereverb_By_FoxJoy":
                pre_fun = MDXNetDereverb(15)
            else:
                func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
                pre_fun = func(
                    agg=int(agg),
                    model_path=model_path,
                    device=device,
                    is_half=is_half,
                )
            
            # 转换格式为44100Hz双声道
            converted_path = os.path.join(tmp_dir, "converted.wav")
            try:
                ffmpeg.input(input_path).output(
                    converted_path,
                    ar=44100,
                    ac=2,
                    acodec="pcm_s16le"
                ).run(overwrite_output=True, quiet=True)
            except ffmpeg.Error as e:
                print(f"FFmpeg error: {e.stderr.decode('utf-8')}")
                raise
            
            # 处理音频
            vocal_path, bgm_path = pre_fun._path_audio_(
                converted_path, save_root_ins, save_root_vocal, formato, is_hp3
            )
            
            # 加载处理后的音频（使用修改后的load_audio_file函数）
            vocal_audio = load_audio_file(vocal_path)
            bgm_audio = load_audio_file(bgm_path)
            
            # 清理
            try:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
                    del pre_fun
            except:
                pass
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return vocal_audio, bgm_audio