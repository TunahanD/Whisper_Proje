# =============================
# FILE: app.py (Yeni Flask Sunucusu)
# =============================
import os
from flask import Flask, request, jsonify
import logging
import time

# Mevcut pipeline modüllerimizi import edelim
# Bu fonksiyon, ses dosyasının byte içeriğini alıp işleyecek olan ana mantığı içeriyor.
from run_pipeline import process_audio_from_file_upload, get_pipeline

# --------------------------------------------------------------------------
# Flask Uygulamasını ve Pipeline'ı Başlatma
# --------------------------------------------------------------------------

# Log ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask uygulamasını oluştur
app = Flask(__name__)

# ÖNEMLİ: Modeli Sadece Bir Kez Yükle!
# Sunucu başlarken, Whisper modelini ve pipeline'ı belleğe yükleyelim.
# Bu sayede her istek geldiğinde modeli tekrar tekrar yüklemek zorunda kalmayız.
# get_pipeline() fonksiyonu singleton yapısı sayesinde bunu bizim için zaten yapıyor.
with app.app_context():
    logging.info("Flask sunucusu başlatılıyor...")
    logging.info("Whisper modeli ve işlem pipeline'ı belleğe yükleniyor. Bu işlem biraz zaman alabilir...")
    start_time = time.time()
    
    # get_pipeline() fonksiyonu, pipeline'ı başlatır ve modeli yükler.
    # Bu fonksiyonu burada çağırmak, sunucu ayağa kalkarken modelin hazır olmasını sağlar.
    get_pipeline() 
    
    end_time = time.time()
    logging.info(f"✅ Model başarıyla yüklendi ve sunucu hazır! (Yükleme süresi: {end_time - start_time:.2f} saniye)")

# --------------------------------------------------------------------------
# API Endpoint Tanımı
# --------------------------------------------------------------------------

@app.route('/process-audio', methods=['POST'])
def process_audio_endpoint():
    """
    Ses dosyasını POST isteği ile alan, işleyen ve niyet analizi sonucunu
    JSON formatında döndüren ana API endpoint'i.
    """
    
    # 1. Gelen isteği kontrol et: 'audio_file' adında bir dosya var mı?
    if 'audio_file' not in request.files:
        logging.warning("İstek 'audio_file' içermiyor.")
        return jsonify({"error": "Lütfen 'audio_file' anahtarıyla bir ses dosyası gönderin."}), 400

    audio_file = request.files['audio_file']

    # 2. Dosya adı var mı kontrol et
    if audio_file.filename == '':
        logging.warning("İstekte dosya seçilmemiş.")
        return jsonify({"error": "Dosya seçilmedi."}), 400

    try:
        logging.info(f"Yeni istek alındı: {audio_file.filename}")
        
        # 3. Dosyanın içeriğini byte olarak oku
        file_content = audio_file.read()
        original_filename = audio_file.filename

        # 4. Ana işlem fonksiyonunu çağır
        # Bu fonksiyon run_pipeline.py içindedir ve tüm süreci yönetir:
        #   - Byte veriyi transcribe eder (FFmpeg ile format dönüşümü dahil)
        #   - Transkripti sözlük düzeltici ile temizler
        #   - Temiz metin üzerinden niyet analizi yapar
        start_process_time = time.time()
        result = process_audio_from_file_upload(file_content, original_filename)
        end_process_time = time.time()
        
        logging.info(f"İşlem tamamlandı: {original_filename} (Süre: {end_process_time - start_process_time:.2f} saniye)")
        
        # 5. Sonucu JSON olarak döndür
        return jsonify(result), 200

    except FileNotFoundError as e:
        logging.error(f"Dosya işlenirken hata: {e}")
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        # Örneğin, transkripsiyon sonucu boş gelirse bu hata fırlatılır
        logging.error(f"Değer hatası: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Beklenmedik diğer tüm hatalar için
        logging.error(f"Sunucuda beklenmedik bir hata oluştu: {e}", exc_info=True)
        return jsonify({"error": "Sunucuda beklenmedik bir hata oluştu. Lütfen logları kontrol edin."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Sunucunun ayakta olup olmadığını kontrol etmek için basit bir endpoint.
    """
    return jsonify({"status": "ok", "message": "Speech-to-Text API çalışıyor."}), 200

# --------------------------------------------------------------------------
# Sunucuyu Çalıştırma
# --------------------------------------------------------------------------

if __name__ == '__main__':
    # Geliştirme sunucusu için:
    # app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    #
    # Windows'ta production için 'waitress' kullanılması önerilir.
    # Aşağıdaki komut satırı talimatlarına bakın.
    print("\n============================================================")
    print("Flask sunucusu geliştirme modunda BAŞLATILAMAZ.")
    print("Lütfen production-ready bir sunucu olan 'waitress' ile başlatın.")
    print("Komut istemine (CMD) veya PowerShell'e şunu yazın:")
    print("\nwaitress-serve --host 0.0.0.0 --port 5000 app:app\n")
    print("============================================================")