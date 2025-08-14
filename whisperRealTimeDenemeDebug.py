import pyaudio
import numpy as np
import whisper
import time

# Basit test - ses seviyelerini ve mikrofonu test edelim
def test_microphone():
    print("=== MİKROFON TESİ ===")
    
    # Ses ayarları
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    audio = pyaudio.PyAudio()
    
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("🎤 Mikrofon çalışıyor. 10 saniye ses seviyesi ölçülecek...")
        print("📢 Konuşmaya başlayın!")
        
        for i in range(100):  # 10 saniye (100 * 0.1)
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            
            print(f"Ses seviyesi: {rms:6.1f} {'🔊' if rms > 100 else '🔇'}", end='\r')
            time.sleep(0.1)
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"❌ Mikrofon hatası: {e}")
    finally:
        audio.terminate()

def test_whisper_simple():
    print("\n\n=== WHİSPER TESİ ===")
    
    # Whisper modelini yükle
    print("Whisper base modeli yükleniyor...")
    model = whisper.load_model("base")
    
    # Ses kaydı
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    
    audio = pyaudio.PyAudio()
    
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print(f"🎤 {RECORD_SECONDS} saniye kayıt başlıyor...")
        print("📢 Şimdi bir şeyler söyleyin!")
        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # İlerleme göster
            progress = (i + 1) / (RATE / CHUNK * RECORD_SECONDS) * 100
            print(f"Kayıt: %{progress:.0f}", end='\r')
        
        print("\n✅ Kayıt tamamlandı!")
        
        # Ses verisini birleştir
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        # Ses seviyesi kontrol
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"🔊 Ortalama ses seviyesi: {rms:.1f}")
        
        if rms < 50:
            print("⚠️  Ses seviyesi çok düşük! Mikrofonu kontrol edin.")
            return
        
        # Whisper için normalize et
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        print("🤖 Whisper ile transkripsiyon yapılıyor...")
        result = model.transcribe(
            audio_float,
            language="tr",
            task="transcribe",
            verbose=True  # Detaylı çıktı
        )
        
        print(f"\n🎯 SONUÇ: {result['text']}")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        audio.terminate()

def main():
    print("🔧 Whisper STT Debug Test")
    print("1. Önce mikrofon test edilecek")
    print("2. Sonra basit Whisper testi yapılacak\n")
    
    # Mikrofon testi
    test_microphone()
    
    input("\nDevam etmek için Enter'a basın...")
    
    # Whisper testi
    test_whisper_simple()

if __name__ == "__main__":
    main()