# =============================
# FILE: main.py (Merkezi CONFIG ile)
# =============================
"""
Terminal tabanlı PTT (F9 başlat, F10 durdur) Türkçe STT - CUDA/GPU sürümü
- faster-whisper (tek seferlik model yükleme; GPU öncelikli)
- sounddevice ile mikrofon seçimi
- Hafif ön işleme: tepe (peak) normalizasyon + opsiyonel gürültü azaltma (noisereduce)
- Kayıt bitince tek seferde transkript (kısa komutlar için optimize)
- Çıktı: ham ve sözlükle düzeltilmiş metin
- Sözlük düzeltici ayrı dosyada: sozluk_duzeltici.py
- Opsiyonel JSON sözlük genişletme: --lex-json terimler.json
- Memory monitoring: 8GB warning, 10GB restart

Notlar:
- Tüm ayarlar **tek bir CONFIG** sözlüğünden yönetilir (aşağıda).
- CLI parametreleri (argparse) sadece bu CONFIG'i override eder.
- CUDA yoksa otomatik CPU/float32 fallback yapılır.
- Memory monitoring her transkripsiyon sonrası çalışır (0 performance impact)
"""

import argparse
import sys
import time
from dataclasses import dataclass
import threading
import numpy as np
import sounddevice as sd
from pynput import keyboard
import gc

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
    # Whisper / Donanım
    "modelSize": "medium",   # small / medium / large-v2 / large-v3
    "device": "cpu",          # "cuda" ya da "cpu"
    "computeType": "int8",  # GPU için "float16", CPU için "float32",CPU da hız için "int8"

    # Ses yakalama
    "targetSr": 16_000,
    "channels": 1,
    "dtype": "float32",        # sounddevice giriş dtype

    # Ön işleme
    "useNoiseReduction": False, # Hızlı çalışmasını istiyorsak False, ön işleme olsun istiyorsak True.
    "nrProfileSec": 0.30, # İlk 300 ms taban gürültü olarak kullanılır.
    "peakTarget": 0.99, # Sesin tepe değerini belirli bir hedef seviyede normalize ediyor.
    
    # Memory monitoring
    "memoryWarningGB": 8.0,    # Warning threshold
    "memoryRestartGB": 10.0,   # Restart threshold
}

# Çalışma boyunca kullanılacak etkin konfigürasyon
CFG = CONFIG.copy()

# =============================
# Global durum
# =============================
whisper_model = None
model_lock = threading.Lock()
memory_monitor = None

@dataclass
class AppState:
    recording: bool = False
    stop_requested: bool = False
    buffer: list = None
    device_index: int = None
    transcription_count: int = 0

state = AppState(recording=False, stop_requested=False, buffer=[], device_index=None, transcription_count=0)

# =============================
# Donanım / Model Yardımcıları
# =============================

def check_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_device_and_compute(device: str, compute_type: str):
    """CUDA yoksa CPU/float32'ye düş."""
    if device == "cuda" and not check_cuda_available():
        print("[Model] Uyarı: CUDA bulunamadı - CPU'ya fallback yapılıyor.")
        device = "cpu"
        if compute_type == "float16":
            compute_type = "float32"
    return device, compute_type


def load_whisper_model(model_size: str = None, device: str = None, compute_type: str = None):
    """Tek seferlik model yükleme. Parametre verilmezse CFG kullanılır."""
    global whisper_model
    if whisper_model is not None:
        return whisper_model

    ms = model_size or CFG["modelSize"]
    dev = device or CFG["device"]
    ct = compute_type or CFG["computeType"]

    dev, ct = resolve_device_and_compute(dev, ct)

    print(f"[Model] faster-whisper yükleniyor -> model={ms}, device={dev}, compute_type={ct} ...")
    try:
        m = WhisperModel(ms, device=dev, compute_type=ct)
    except Exception as e:
        print(f"[Model] Hata: model yüklenemedi: {e}")
        print("[Model] CPU fallback deneniyor (compute_type=float32)")
        m = WhisperModel(ms, device="cpu", compute_type="float32")

    whisper_model = m
    print("[Model] ✅ Model yüklendi ve hazır.")
    return whisper_model

# =============================
# Mikrofon listesi/Seçimi
# =============================

def listele_ve_sec_mikrofon() -> int:
    print("\n=== Ses Giriş Aygıtları ===")
    devices = sd.query_devices()
    input_indices = []
    for idx, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) > 0:
            input_indices.append(idx)
            print(f"[{idx}] {dev['name']} (in:{dev['max_input_channels']} out:{dev['max_output_channels']})")
    if not input_indices:
        print("Giriş aygıtı bulunamadı! Lütfen Windows ses ayarlarını kontrol edin.")
        sys.exit(1)
    while True:
        try:
            sel = int(input("\nKullanmak istediğiniz mikrofon index'i: "))
            if sel in input_indices:
                return sel
            else:
                print("Geçersiz seçim. Yalnızca listelenen index'lerden birini girin.")
        except ValueError:
            print("Lütfen sayısal bir index girin.")

# =============================
# PTT (F9/F10) dinleyicisi
# =============================

def on_press(key):
    try:
        if key == keyboard.Key.f9 and not state.recording:
            print("\n[F9] 🎤 Kayıt BAŞLADI. [F10] ile durdurabilirsiniz…")
            state.recording = True
            state.stop_requested = False
            state.buffer = []
        elif key == keyboard.Key.f10 and state.recording:
            print("\n[F10] ⏹️ Kayıt DURDURULDU. İşleniyor…")
            state.stop_requested = True
        elif key == keyboard.Key.f1:  # Memory stats için F1 ekledim
            if memory_monitor:
                memory_monitor.print_memory_stats()
    except Exception as e:
        print(f"Klavye dinleyicisi hatası: {e}")


def klavye_dinleyici_baslat():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener

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
        print(f"[Uyarı] Gürültü azaltma sırasında hata: {e}. Atlandı.")
        return x

# =============================
# Ses akışı callback'i
# =============================

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[SoundDevice] Durum: {status}")
    if state.recording and not state.stop_requested:
        # mono
        mono = np.mean(indata, axis=1).astype(np.float32)
        state.buffer.append(mono)

# =============================
# Memory temizliği ve monitoring
# =============================

def cleanup_after_transcription():
    """Transkripsiyon sonrası memory temizliği"""
    # Manuel garbage collection
    gc.collect()
    
    # CUDA memory temizliği (varsa)
    if CFG["device"] == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def check_memory_and_restart_if_needed():
    """Memory monitoring ve gerekirse restart"""
    global memory_monitor
    
    if memory_monitor is None:
        return
    
    # Memory kontrolü yap (0.1ms overhead)
    if memory_monitor.should_restart():
        print("\n[Memory Monitor] Critical memory usage detected!")
        print("[Memory Monitor] Restarting program...")
        
        # Cleanup before restart
        try:
            cleanup_after_transcription()
        except Exception:
            pass
        
        # Program restart
        restart_program()

# =============================
# Transkripsiyon
# =============================

def transcribe_numpy(audio_np: np.ndarray, sr: int, lex_json: str = None):
    """audio_np: float32 mono"""
    # Ön işleme
    audio_np = peak_normalize(audio_np)
    audio_np = noise_reduce_if_needed(audio_np, sr)

    # Modeli al (tek sefer yüklenir)
    with model_lock:
        model = load_whisper_model()

    print("[STT] ⚡ Çözümleme başlıyor…")
    segments, info = model.transcribe(audio_np, language="tr", task="transcribe") # Hızlanması için => beam_size=1,vad_filter=True

    raw_text_parts = [seg.text for seg in segments]
    raw_text = " ".join(s.strip() for s in raw_text_parts).strip()

    corrected = sozlukDuzelt2(raw_text)

    print("\n==== SONUÇ ====")
    print("Ham metin:", raw_text)
    print("Düzeltilmiş metin:", corrected)
    print("================")
    
    # Memory temizliği (segments ve info objelerini temizle)
    try:
        del segments, info, raw_text_parts
    except Exception:
        pass
    
    
    # Post-processing: Memory monitoring (0 performance impact)
    cleanup_after_transcription()
    check_memory_and_restart_if_needed()
    

# =============================
# Ana döngü (CLI)
# =============================

def main():
    global memory_monitor
    
    parser = argparse.ArgumentParser(description="PTT Türkçe STT (faster-whisper, GPU-ready)")
    # Argparse varsayılanlarını None bırakıp sadece override için kullanıyoruz
    parser.add_argument('--model-size', type=str, default=None, help="Model boyutu (small/medium/large-v2/large-v3)")
    parser.add_argument('--device', type=str, default=None, help="'cuda' veya 'cpu'")
    parser.add_argument('--compute-type', type=str, default=None, help="compute_type (float16, int8, float32)")
    parser.add_argument('--lex-json', type=str, default=None, help='Opsiyonel sözlük JSON dosyası')
    parser.add_argument('--no-nr', action='store_true', help='Gürültü azaltmayı devre dışı bırak')
    parser.add_argument('--no-memory-monitor', action='store_true', help='Memory monitoring devre dışı bırak')
    parser.add_argument('--memory-warning', type=float, default=None, help='Memory warning threshold (GB)')
    parser.add_argument('--memory-restart', type=float, default=None, help='Memory restart threshold (GB)')
    args = parser.parse_args()

    # CONFIG override
    if args.model_size:
        CFG["modelSize"] = args.model_size
    if args.device:
        CFG["device"] = args.device
    if args.compute_type:
        CFG["computeType"] = args.compute_type
    if args.no_nr:
        CFG["useNoiseReduction"] = False
    if args.memory_warning:
        CFG["memoryWarningGB"] = args.memory_warning
    if args.memory_restart:
        CFG["memoryRestartGB"] = args.memory_restart

    # Memory Monitor başlat (opsiyonel)
    if not args.no_memory_monitor:
        try:
            memory_monitor = MemoryMonitor(
                warning_threshold_gb=CFG["memoryWarningGB"],
                restart_threshold_gb=CFG["memoryRestartGB"]
            )
        except Exception as e:
            print(f"[Memory Monitor] Hata: {e}")
            print("[Memory Monitor] Memory monitoring devre dışı kalacak")
            memory_monitor = None

    # JSON sözlük varsa program başında yükle
    if args.lex_json:
        if yukle_json_ve_birlestir is None:
            print("[Sözlük] Uyarı: 'yukle_json_ve_birlestir' fonksiyonu bulunamadı. sozluk_duzeltici.py sürümünü kontrol et.")
        else:
            try:
                yukle_json_ve_birlestir(args.lex_json)
                print(f"[Sözlük] JSON sözlük yüklendi: {args.lex_json}")
            except Exception as e:
                print(f"[Sözlük] JSON yükleme hatası: {e}")

    # Mikrofon seçimi
    try:
        state.device_index = listele_ve_sec_mikrofon()
    except Exception as e:
        print(f"Mikrofon listelenemedi: {e}")
        sys.exit(1)

    # Program başında modeli yüklemeyi dene (hata yakala ama devam et)
    try:
        load_whisper_model()
    except Exception as e:
        print(f"[Model Başlatma] Hata: {e}")

    # Bilgi bandosu
    eff_dev, eff_ct = resolve_device_and_compute(CFG["device"], CFG["computeType"])
    print("\n" + "="*50)
    print("✅ SİSTEM HAZIR! (GPU-ready + Memory Monitor)")
    print(f"• Model: {CFG['modelSize']}")
    print(f"• Device/Compute: {eff_dev} / {eff_ct}")
    print(f"• Audio: {CFG['targetSr']} Hz, {CFG['channels']} ch, dtype={CFG['dtype']}")
    print(f"• NR: {'Açık' if CFG['useNoiseReduction'] else 'Kapalı'} (profil {CFG['nrProfileSec']} sn)")
    if memory_monitor:
        print(f"• Memory: Warning {CFG['memoryWarningGB']}GB, Restart {CFG['memoryRestartGB']}GB")
    print("🎤 [F9] ile kaydı başlatın")
    print("⏹️  [F10] ile kayıt durdurun")
    print("📊 [F1] ile memory stats")
    print("❌ Çıkış için Ctrl+C")
    print("="*50)

    klavye_dinleyici_baslat()

    try:
        with sd.InputStream(
            samplerate=CFG["targetSr"],
            channels=CFG["channels"],
            dtype=CFG["dtype"],
            device=state.device_index,
            callback=audio_callback,
            blocksize=0,
        ):
            while True:
                time.sleep(0.05)
                if state.recording and state.stop_requested:
                    audio = np.concatenate(state.buffer, axis=0).astype(np.float32) if state.buffer else np.zeros((0,), dtype=np.float32)
                    state.recording = False
                    state.stop_requested = False
                    state.buffer = []

                    if len(audio) > 0:
                        duration = len(audio) / CFG["targetSr"]
                        print(f"[Kayıt] {duration:.2f} saniye ses yakalandı.")
                        if duration > 0.1:  # Minimum süre kontrolü
                            transcribe_numpy(audio, CFG["targetSr"], args.lex_json)
                        else:
                            print("[Kayıt] Çok kısa ses kaydı, atlandı.")
                    else:
                        print("[Kayıt] Ses verisi yakalanmadı.")
                    
                    print("\n🎤 [F9] ile yeni kayda başlayabilirsiniz...")

    except KeyboardInterrupt:
        print("\n[Çıkış] Program sonlandırılıyor...")
    except Exception as e:
        print(f"\n[Hata] Ses akışı hatası: {e}")
    finally:
        print("Program kapatıldı.")


if __name__ == "__main__":
    main()