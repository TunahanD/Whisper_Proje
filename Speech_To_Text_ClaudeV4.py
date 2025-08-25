# =============================
# FILE: main.py (WAV Transcribe)
# =============================
"""
Dosya tabanlı Türkçe STT - CUDA/GPU sürümü
- faster-whisper (tek seferlik model yükleme; GPU öncelikli)
- .wav dosyasını oku -> numpy -> ön işleme -> transkript
- Çıktı: ham ve sözlükle düzeltilmiş metin
- Sözlük düzeltici ayrı dosyada: sozluk_duzeltici.py
- Opsiyonel JSON sözlük genişletme: --lex-json terimler.json
- Memory monitoring: 8GB warning, 10GB restart
"""

import argparse
import sys
import numpy as np
import gc
import soundfile as sf  # WAV okumak için

from faster_whisper import WhisperModel

# Sözlük düzeltici modülü
from sozluk_duzeltici import sozlukDuzelt2
try:
    from sozluk_duzeltici import yukle_json_ve_birlestir
except Exception:
    yukle_json_ve_birlestir = None

# Memory monitoring modülü
from memory_monitor import MemoryMonitor, restart_program

# =============================
# CONFIG (TEK MERKEZ)
# =============================
CONFIG = {
    # Ses dosyası yolu (buradan değiştirilebilir)
    "audioFile": "Kayıt.wav",  # WAV dosyasının yolu
    
    "modelSize": "medium",     # small / medium / large-v2 / large-v3
    "device": "cuda",          # "cuda" ya da "cpu"
    "computeType": "float16",  # GPU için "float16", CPU için "int8"/"float32"

    # Ses parametreleri
    "targetSr": 16_000,
    "channels": 1,
    "dtype": "float32",

    # Ön işleme
    "useNoiseReduction": False,
    "nrProfileSec": 0.30,
    "peakTarget": 0.99,
    
    # Memory monitoring
    "memoryWarningGB": 8.0,
    "memoryRestartGB": 10.0,
}

CFG = CONFIG.copy()

# =============================
# Global durum
# =============================
whisper_model = None
memory_monitor = None

# =============================
# Yardımcılar
# =============================
def check_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def resolve_device_and_compute(device: str, compute_type: str):
    if device == "cuda" and not check_cuda_available():
        print("[Model] Uyarı: CUDA bulunamadı - CPU'ya fallback yapılıyor.")
        device = "cpu"
        if compute_type == "float16":
            compute_type = "float32"
    return device, compute_type

def load_whisper_model():
    global whisper_model
    if whisper_model is not None:
        return whisper_model

    ms = CFG["modelSize"]
    dev, ct = resolve_device_and_compute(CFG["device"], CFG["computeType"])

    print(f"[Model] faster-whisper yükleniyor -> model={ms}, device={dev}, compute_type={ct} ...")
    try:
        m = WhisperModel(ms, device=dev, compute_type=ct)
    except Exception as e:
        print(f"[Model] Hata: {e}, CPU fallback deneniyor...")
        m = WhisperModel(ms, device="cpu", compute_type="float32")

    whisper_model = m
    print("[Model] ✅ Model yüklendi ve hazır.")
    return whisper_model

# =============================
# Ön işleme
# =============================
def peak_normalize(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-12
    if peak > 0:
        gain = min(CFG["peakTarget"] / peak, 10.0)
        x = x * gain
    return x

def noise_reduce_if_needed(x: np.ndarray, sr: int) -> np.ndarray:
    if not CFG["useNoiseReduction"]:
        return x
    try:
        import noisereduce as nr
    except ImportError:
        print("[Uyarı] noisereduce kurulu değil, gürültü azaltma atlandı.")
        return x
    n_prof = int(CFG["nrProfileSec"] * sr)
    noise_prof = x[:n_prof] if len(x) > n_prof else x
    try:
        y = nr.reduce_noise(y=x, sr=sr, y_noise=noise_prof, stationary=True)
        return y.astype(np.float32)
    except Exception as e:
        print(f"[Uyarı] Gürültü azaltma hatası: {e}. Atlandı.")
        return x

# =============================
# Memory temizliği
# =============================
def cleanup_after_transcription():
    gc.collect()
    if CFG["device"] == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def check_memory_and_restart_if_needed():
    global memory_monitor
    if memory_monitor and memory_monitor.should_restart():
        print("\n[Memory Monitor] Critical usage! Restarting...")
        cleanup_after_transcription()
        restart_program()

# =============================
# Transkripsiyon
# =============================
def transcribe_wav(file_path: str):
    # WAV oku
    audio_np, sr = sf.read(file_path, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)  # stereo -> mono
    
    # SR kontrol
    if sr != CFG["targetSr"]:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=CFG["targetSr"])
        sr = CFG["targetSr"]

    # Ön işleme
    audio_np = peak_normalize(audio_np)
    audio_np = noise_reduce_if_needed(audio_np, sr)

    # Modeli al
    model = load_whisper_model()

    print(f"[STT] ⚡ Çözümleme başlıyor: {file_path}")
    segments, info = model.transcribe(audio_np, language="tr", task="transcribe")

    raw_text = " ".join(seg.text.strip() for seg in segments).strip()
    corrected = sozlukDuzelt2(raw_text)

    print("\n==== SONUÇ ====")
    print("Ham metin:", raw_text)
    print("Düzeltilmiş metin:", corrected)
    print("================")

    del segments, info
    cleanup_after_transcription()
    check_memory_and_restart_if_needed()

# =============================
# Ana CLI
# =============================
def main():
    global memory_monitor
    parser = argparse.ArgumentParser(description="WAV Türkçe STT (faster-whisper, GPU-ready)")
    parser.add_argument('--wav', type=str, default=None, help='Transkribe edilecek wav dosyası (opsiyonel, CONFIG\'ten de ayarlanabilir)')
    parser.add_argument('--model-size', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--compute-type', type=str, default=None)
    parser.add_argument('--lex-json', type=str, default=None)
    parser.add_argument('--no-nr', action='store_true')
    parser.add_argument('--no-memory-monitor', action='store_true')
    parser.add_argument('--memory-warning', type=float, default=None)
    parser.add_argument('--memory-restart', type=float, default=None)
    args = parser.parse_args()

    # Override config - Ses dosyası yolu kontrolü
    if args.wav:  # CLI'dan dosya yolu verilmişse CONFIG'i override et
        CFG["audioFile"] = args.wav
    
    if args.model_size: CFG["modelSize"] = args.model_size
    if args.device: CFG["device"] = args.device
    if args.compute_type: CFG["computeType"] = args.compute_type
    if args.no_nr: CFG["useNoiseReduction"] = False
    if args.memory_warning: CFG["memoryWarningGB"] = args.memory_warning
    if args.memory_restart: CFG["memoryRestartGB"] = args.memory_restart

    # Memory monitor
    if not args.no_memory_monitor:
        try:
            memory_monitor = MemoryMonitor(
                warning_threshold_gb=CFG["memoryWarningGB"],
                restart_threshold_gb=CFG["memoryRestartGB"]
            )
        except Exception as e:
            print(f"[Memory Monitor] Hata: {e}")
            memory_monitor = None

    # JSON sözlük
    if args.lex_json and yukle_json_ve_birlestir:
        try:
            yukle_json_ve_birlestir(args.lex_json)
            print(f"[Sözlük] JSON sözlük yüklendi: {args.lex_json}")
        except Exception as e:
            print(f"[Sözlük] JSON yükleme hatası: {e}")

    # Bilgi bandosu
    dev, ct = resolve_device_and_compute(CFG["device"], CFG["computeType"])
    print("\n" + "="*50)
    print("✅ WAV TRANSKRIPSIYON SİSTEMİ HAZIR!")
    print(f"• Audio File: {CFG['audioFile']}")
    print(f"• Model: {CFG['modelSize']}")
    print(f"• Device/Compute: {dev} / {ct}")
    print(f"• Target SR: {CFG['targetSr']} Hz")
    print(f"• NR: {'Açık' if CFG['useNoiseReduction'] else 'Kapalı'}")
    if memory_monitor:
        print(f"• Memory: Warning {CFG['memoryWarningGB']}GB, Restart {CFG['memoryRestartGB']}GB")
    print("="*50)

    # Transcribe (CONFIG'ten dosya yolunu al)
    transcribe_wav(CFG["audioFile"])

if __name__ == "__main__":
    main()