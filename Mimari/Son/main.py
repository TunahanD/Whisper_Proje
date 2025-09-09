import argparse, sys, numpy as np, gc, soundfile as sf, subprocess, tempfile, os
from faster_whisper import WhisperModel
from pathlib import Path
import logging, shlex
from sozluk_duzeltici import sozlukDuzelt2
from memory_monitor import MemoryMonitor, restart_program

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Varsayılan konfigürasyonu CPU için optimize ettim
DEFAULT_CONFIG = {
    "modelSize": "large-v3", 
    "device": "cpu", 
    "computeType": "int8", # CPU için en hızlı seçeneklerden biri
    "targetSr": 16_000,
    "channels": 1, "dtype": "float32", "useNoiseReduction": False, "nrProfileSec": 0.30,
    "peakTarget": 0.99, "memoryWarningGB": 8.0, "memoryRestartGB": 12.0,
    "ffmpeg_path": "ffmpeg", "supported_formats": [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".webm"],
    "temp_dir": None,
}

class TranscriptionEngine:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.memory_monitor = MemoryMonitor(warning_threshold_gb=self.config["memoryWarningGB"], restart_threshold_gb=self.config["memoryRestartGB"])

    def load_model(self):
        if self.model is not None: return self.model
        ms = self.config["modelSize"]
        dev = self.config["device"]
        ct = self.config["computeType"]
        
        logger.info(f"faster-whisper modeli yükleniyor -> model={ms}, device={dev}, compute_type={ct}")
        try:
            self.model = WhisperModel(ms, device=dev, compute_type=ct)
        except Exception as e:
            logger.error(f"Model yüklenemedi: {e}")
            raise e # Hata varsa program dursun
            
        logger.info("✅ Model yüklendi ve hazır.")
        return self.model

    def _convert_to_wav_with_ffmpeg(self, input_path: str) -> str:
        temp_dir = self.config["temp_dir"] or tempfile.gettempdir()
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', dir=temp_dir, delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        ffmpeg_cmd = [self.config["ffmpeg_path"], "-i", input_path, "-ar", str(self.config["targetSr"]), "-ac", "1", "-c:a", "pcm_s16le", "-y", temp_wav_path]
        try:
            logger.info(f"FFmpeg dönüşümü başlıyor: {Path(input_path).name} -> WAV")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                error_details = f"FFmpeg işlemi başarısız. STDERR: {result.stderr.strip()}"
                logger.error(error_details)
                raise subprocess.SubprocessError(error_details)
            return temp_wav_path
        except Exception as e:
            logger.error(f"FFmpeg dönüştürmede beklenmedik bir hata oluştu: {e}", exc_info=True)
            if os.path.exists(temp_wav_path): os.unlink(temp_wav_path)
            raise e

    def transcribe_file(self, file_path: str) -> dict:
        if self.model is None: self.load_model()
        original_path, temp_wav_path = file_path, None
        try:
            if Path(file_path).suffix.lower() not in ['.wav', '.pcm']:
                temp_wav_path = self._convert_to_wav_with_ffmpeg(file_path)
                file_path = temp_wav_path
            
            audio_np, sr = sf.read(file_path, dtype="float32")
            if audio_np.ndim > 1: audio_np = np.mean(audio_np, axis=1)
            
            logger.info(f"⚡ Çözümleme başlıyor: {Path(original_path).name}")
            segments, _ = self.model.transcribe(audio_np, language="tr", task="transcribe")
            raw_text = " ".join(seg.text.strip() for seg in segments).strip()
            corrected_text = sozlukDuzelt2(raw_text)
            
            return {"raw_text": raw_text, "corrected_text": corrected_text}
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except Exception as e: 
                    logger.warning(f"Geçici dosya silinemedi: {e}")

    def transcribe_from_bytes(self, audio_bytes: bytes, original_filename: str = "audio.webm") -> dict:
        file_ext = Path(original_filename).suffix.lower() or ".webm"
        temp_dir = self.config["temp_dir"] or tempfile.gettempdir()
        temp_input = tempfile.NamedTemporaryFile(suffix=file_ext, dir=temp_dir, delete=False)
        try:
            temp_input.write(audio_bytes)
            temp_input.close()
            return self.transcribe_file(temp_input.name)
        finally:
            if os.path.exists(temp_input.name):
                try: 
                    os.unlink(temp_input.name)
                except Exception as e: 
                    logger.warning(f"Geçici input dosyası silinemedi: {e}")