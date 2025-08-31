# =============================
# FILE: main.py (WAV Transcribe) - REFACTORED
# =============================
"""
Dosya tabanlı Türkçe STT - CUDA/GPU sürümü (Yeniden düzenlenmiş)
- faster-whisper (tek seferlik model yükleme; GPU öncelikli)
- .wav dosyasını oku -> numpy -> ön işleme -> transkript
- Modüler yapı: CLI ve programmatic kullanım ayrı
- Memory monitoring entegreli
"""

import argparse
import sys
import numpy as np
import gc
import soundfile as sf
from faster_whisper import WhisperModel
import subprocess
import tempfile
import os
from pathlib import Path

# Sözlük düzeltici modülü
from sozluk_duzeltici import sozlukDuzelt2
try:
    from sozluk_duzeltici import yukle_json_ve_birlestir
except Exception:
    yukle_json_ve_birlestir = None

# Memory monitoring modülü
from memory_monitor import MemoryMonitor, restart_program

# =============================
# CONFIG (Sadece model ayarları)
# =============================
DEFAULT_CONFIG = {
    "modelSize": "medium",
    "device": "cpu", 
    "computeType": "float32",
    "targetSr": 16_000,
    "channels": 1,
    "dtype": "float32",
    "useNoiseReduction": False,
    "nrProfileSec": 0.30,
    "peakTarget": 0.99,
    "memoryWarningGB": 8.0,
    "memoryRestartGB": 10.0,
    # FFmpeg ayarları
    "ffmpeg_path": "ffmpeg",  # System PATH'de ffmpeg varsa
    "supported_formats": [".wav", ".mp3", ".mp4", ".m4a", ".flac", ".aac", ".ogg", ".webm"],
    "temp_dir": None,  # None ise sistem temp dizini kullanılır
}

# =============================
# TranscriptionEngine Class
# =============================
class TranscriptionEngine:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = None
        
        # Memory monitor'ı başlat
        self.memory_monitor = MemoryMonitor(
            warning_threshold_gb=self.config["memoryWarningGB"],
            restart_threshold_gb=self.config["memoryRestartGB"]
        )
        
    def _check_cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _resolve_device_and_compute(self, device: str, compute_type: str):
        if device == "cuda" and not self._check_cuda_available():
            print("[Model] Uyarı: CUDA bulunamadı - CPU'ya fallback yapılıyor.")
            device = "cpu"
            if compute_type == "float16":
                compute_type = "float32"
        return device, compute_type

    def load_model(self):
        """Model yükleme - sadece bir kez çalıştırılır"""
        if self.model is not None:
            return self.model

        ms = self.config["modelSize"]
        dev, ct = self._resolve_device_and_compute(
            self.config["device"], 
            self.config["computeType"]
        )

        print(f"[Model] faster-whisper yükleniyor -> model={ms}, device={dev}, compute_type={ct}")
        try:
            self.model = WhisperModel(ms, device=dev, compute_type=ct)
        except Exception as e:
            print(f"[Model] Hata: {e}, CPU fallback deneniyor...")
            self.model = WhisperModel(ms, device="cpu", compute_type="float32")

        print("[Model] ✅ Model yüklendi ve hazır.")
        
        # Model yüklendikten sonra memory check
        memory_status = self.memory_monitor.check_memory_threshold()
        if memory_status == 'restart':
            print("[Memory] Model yükleme sonrası restart gerekiyor...")
            restart_program()
        
        return self.model

    def _check_ffmpeg_available(self) -> bool:
        """FFmpeg'in sistem üzerinde kurulu olup olmadığını kontrol eder"""
        try:
            result = subprocess.run(
                [self.config["ffmpeg_path"], "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _convert_to_wav_with_ffmpeg(self, input_path: str) -> str:
        """
        FFmpeg kullanarak herhangi bir ses formatını WAV'a dönüştürür
        Returns: Geçici WAV dosyasının yolu
        """
        if not self._check_ffmpeg_available():
            raise RuntimeError(
                f"FFmpeg bulunamadı! Lütfen ffmpeg'i kurun veya "
                f"config['ffmpeg_path'] ayarını kontrol edin."
            )

        # Geçici WAV dosyası oluştur
        temp_dir = self.config["temp_dir"] or tempfile.gettempdir()
        temp_wav = tempfile.NamedTemporaryFile(
            suffix='.wav',
            dir=temp_dir,
            delete=False
        )
        temp_wav_path = temp_wav.name
        temp_wav.close()

        # FFmpeg komutunu oluştur
        ffmpeg_cmd = [
            self.config["ffmpeg_path"],
            "-i", input_path,  # Input dosyası
            "-ar", str(self.config["targetSr"]),  # Sample rate
            "-ac", "1",  # Mono (1 channel)
            "-c:a", "pcm_s16le",  # WAV codec
            "-y",  # Overwrite output
            temp_wav_path
        ]

        try:
            print(f"[FFmpeg] Dönüştürme başlıyor: {Path(input_path).name} -> WAV")
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 dakika timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Bilinmeyen FFmpeg hatası"
                raise subprocess.SubprocessError(f"FFmpeg hatası: {error_msg}")
            
            print("[FFmpeg] ✅ Dönüştürme başarılı")
            return temp_wav_path
            
        except subprocess.TimeoutExpired:
            # Geçici dosyayı temizle
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            raise RuntimeError("FFmpeg dönüştürme işlemi timeout'a uğradı (5dk)")
        
        except Exception as e:
            # Geçici dosyayı temizle
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            raise RuntimeError(f"FFmpeg dönüştürme hatası: {e}")

    def _detect_audio_format(self, file_path: str) -> str:
        """Dosya uzantısından format tespit eder"""
        return Path(file_path).suffix.lower()

    def _is_supported_format(self, file_path: str) -> bool:
        """Desteklenen format olup olmadığını kontrol eder"""
        ext = self._detect_audio_format(file_path)
        return ext in self.config["supported_formats"]

    def _peak_normalize(self, x: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(x)) + 1e-12
        if peak > 0:
            gain = min(self.config["peakTarget"] / peak, 10.0)
            x = x * gain
        return x

    def _noise_reduce_if_needed(self, x: np.ndarray, sr: int) -> np.ndarray:
        if not self.config["useNoiseReduction"]:
            return x
        try:
            import noisereduce as nr
            n_prof = int(self.config["nrProfileSec"] * sr)
            noise_prof = x[:n_prof] if len(x) > n_prof else x
            y = nr.reduce_noise(y=x, sr=sr, y_noise=noise_prof, stationary=True)
            return y.astype(np.float32)
        except ImportError:
            print("[Uyarı] noisereduce kurulu değil, gürültü azaltma atlandı.")
            return x
        except Exception as e:
            print(f"[Uyarı] Gürültü azaltma hatası: {e}. Atlandı.")
            return x

    def _cleanup(self):
        """Memory temizleme - artık memory monitor ile entegre"""
        gc.collect()
        if self.config["device"] == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        
        # Memory monitor check - temizlik sonrası
        memory_status = self.memory_monitor.check_memory_threshold()
        if memory_status == 'restart':
            print("[Memory] Cleanup sonrası restart gerekiyor...")
            restart_program()

    def transcribe_file(self, file_path: str) -> dict:
        """
        Dosya yolundan transkripsiyon yapar (herhangi bir ses formatını destekler)
        Returns: {"raw_text": str, "corrected_text": str}
        """
        # Model kontrolü
        if self.model is None:
            self.load_model()

        # Format kontrolü ve dönüştürme
        original_path = file_path
        temp_wav_path = None
        
        try:
            # Desteklenen format kontrolü
            if not self._is_supported_format(file_path):
                file_ext = self._detect_audio_format(file_path)
                raise ValueError(
                    f"Desteklenmeyen format: {file_ext}. "
                    f"Desteklenen formatlar: {', '.join(self.config['supported_formats'])}"
                )
            
            # WAV değilse FFmpeg ile dönüştür
            if self._detect_audio_format(file_path) != '.wav':
                print(f"[Format] {Path(file_path).name} WAV formatına dönüştürülüyor...")
                temp_wav_path = self._convert_to_wav_with_ffmpeg(file_path)
                file_path = temp_wav_path
            
            # WAV oku
            audio_np, sr = sf.read(file_path, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)

            # SR kontrol (FFmpeg zaten hedef SR'ye dönüştürür ama double-check)
            if sr != self.config["targetSr"]:
                import librosa
                print(f"[Audio] Sample rate dönüştürme: {sr} -> {self.config['targetSr']} Hz")
                audio_np = librosa.resample(
                    audio_np, 
                    orig_sr=sr, 
                    target_sr=self.config["targetSr"]
                )

            # Ön işleme
            audio_np = self._peak_normalize(audio_np)
            audio_np = self._noise_reduce_if_needed(audio_np, self.config["targetSr"])

            # Transkripsiyon öncesi memory check
            memory_status = self.memory_monitor.check_memory_threshold()
            if memory_status == 'restart':
                print("[Memory] Transkripsiyon öncesi restart gerekiyor...")
                restart_program()

            # Transkripsiyon
            print(f"[STT] ⚡ Çözümleme başlıyor: {Path(original_path).name}")
            segments, info = self.model.transcribe(
                audio_np, 
                language="tr", 
                task="transcribe"
            )

            raw_text = " ".join(seg.text.strip() for seg in segments).strip()
            corrected_text = sozlukDuzelt2(raw_text)

            # Temizlik
            del segments, info
            self._cleanup()  # Memory monitor check bu fonksiyonun içinde

            # Transkripsiyon sonrası final memory check
            memory_status = self.memory_monitor.check_memory_threshold()
            if memory_status == 'restart':
                print("[Memory] Transkripsiyon sonrası restart gerekiyor...")
                restart_program()

            return {
                "raw_text": raw_text,
                "corrected_text": corrected_text,
                "original_file": original_path,
                "processed_file": file_path
            }
            
        finally:
            # Geçici WAV dosyasını temizle
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                    print(f"[Cleanup] Geçici dosya silindi: {Path(temp_wav_path).name}")
                except Exception as e:
                    print(f"[Cleanup] Geçici dosya silinemedi: {e}")

    def transcribe_from_bytes(self, audio_bytes: bytes, original_filename: str = "audio") -> dict:
        """
        Byte array'den transkripsiyon yapar (Flask için)
        audio_bytes: Ses dosyasının byte içeriği
        original_filename: Orijinal dosya adı (format tespiti için)
        """
        # Memory check - işlem öncesi
        memory_status = self.memory_monitor.check_memory_threshold()
        if memory_status == 'restart':
            print("[Memory] Byte transcription öncesi restart gerekiyor...")
            restart_program()
        
        # Geçici dosya oluştur
        file_ext = Path(original_filename).suffix.lower()
        if not file_ext:
            file_ext = ".wav"  # Default format
        
        temp_dir = self.config["temp_dir"] or tempfile.gettempdir()
        temp_input = tempfile.NamedTemporaryFile(
            suffix=file_ext,
            dir=temp_dir,
            delete=False
        )
        
        try:
            # Byte'ları dosyaya yaz
            temp_input.write(audio_bytes)
            temp_input.close()
            
            # Normal transcribe_file fonksiyonunu kullan (memory monitoring dahil)
            result = self.transcribe_file(temp_input.name)
            result["original_filename"] = original_filename
            
            return result
            
        finally:
            # Geçici input dosyasını temizle
            if os.path.exists(temp_input.name):
                try:
                    os.unlink(temp_input.name)
                except Exception as e:
                    print(f"[Cleanup] Geçici input dosyası silinemedi: {e}")

    def get_memory_stats(self) -> dict:
        """Memory istatistiklerini döndür (debug/monitoring için)"""
        return self.memory_monitor.get_memory_stats()
    
    def print_memory_stats(self):
        """Memory istatistiklerini yazdır (debug için)"""
        self.memory_monitor.print_memory_stats()

# =============================
# Global engine instance
# =============================
_engine = None

def get_engine():
    """Singleton pattern - tek bir engine instance'ı döndürür"""
    global _engine
    if _engine is None:
        _engine = TranscriptionEngine()
    return _engine

# =============================
# Public API (Backwards compatibility)
# =============================
def transcribe_wav(file_path: str) -> str:
    """
    Eski API ile uyumluluk için
    Returns: corrected_text
    """
    engine = get_engine()
    result = engine.transcribe_file(file_path)
    return result["corrected_text"]

def load_whisper_model():
    """Eski API ile uyumluluk için"""
    engine = get_engine()
    return engine.load_model()

# =============================
# CLI Interface
# =============================
def main():
    parser = argparse.ArgumentParser(description="WAV Türkçe STT")
    parser.add_argument('wav_file', type=str, help='Transkribe edilecek WAV dosyası')
    parser.add_argument('--model-size', type=str, default="medium")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--compute-type', type=str, default="float32")
    parser.add_argument('--json-output', action='store_true', help='JSON formatında çıktı')
    parser.add_argument('--memory-stats', action='store_true', help='Memory istatistiklerini göster')
    
    args = parser.parse_args()

    # Config oluştur
    config = {
        "modelSize": args.model_size,
        "device": args.device,
        "computeType": args.compute_type
    }

    # Engine oluştur ve transkripsiyon yap
    engine = TranscriptionEngine(config)
    
    # Memory stats istendiyse göster
    if args.memory_stats:
        engine.print_memory_stats()
    
    result = engine.transcribe_file(args.wav_file)

    if args.json_output:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n==== SONUÇ ====")
        print("Ham metin:", result["raw_text"])
        print("Düzeltilmiş metin:", result["corrected_text"])
        print("================")

    # Final memory stats (sadece --memory-stats ile)
    if args.memory_stats:
        print("\n=== Final Memory Stats ===")
        engine.print_memory_stats()

if __name__ == "__main__":
    main()