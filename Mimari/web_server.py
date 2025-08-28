from flask import Flask, request, jsonify
from io import BytesIO
import os
from werkzeug.utils import secure_filename

# Import ettiğimiz modüller
from run_pipeline import process_audio_from_memory

# Flask uygulaması
app = Flask(__name__)

# Konfigürasyon
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max dosya boyutu

# Desteklenen dosya formatları
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def health_check():
    """API çalışıyor mu kontrolü"""
    return jsonify({
        "status": "OK",
        "message": "Speech-to-Text API is running",
        "version": "1.0"
    })

@app.route('/process', methods=['POST'])
def process_audio():
    """Ana endpoint - ses dosyası işleme"""
    try:
        # Dosya kontrolü
        if 'audio_file' not in request.files:
            return jsonify({
                "error": "No audio file provided",
                "message": "Please upload a file with key 'audio_file'"
            }), 400
        
        audio_file = request.files['audio_file']
        
        # Boş dosya kontrolü
        if audio_file.filename == '':
            return jsonify({
                "error": "No file selected",
                "message": "Please select a valid audio file"
            }), 400
        
        # Dosya formatı kontrolü
        if not allowed_file(audio_file.filename):
            return jsonify({
                "error": "Invalid file format",
                "message": f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Dosyayı RAM'de BytesIO'ya dönüştür
        audio_bytes = BytesIO()
        audio_file.save(audio_bytes)
        audio_bytes.seek(0)  # Başa dön
        
        print(f"📁 Dosya alındı: {audio_file.filename} ({len(audio_bytes.getvalue())} bytes)")
        
        # Pipeline'ı çalıştır
        intent_result = process_audio_from_memory(audio_bytes)
        
        # Sonucu döndür
        return jsonify({
            "status": "success",
            "filename": secure_filename(audio_file.filename),
            "result": intent_result
        }), 200
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return jsonify({
            "error": "Processing failed",
            "message": str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Dosya çok büyük hatası"""
    return jsonify({
        "error": "File too large",
        "message": "Maximum file size is 16MB"
    }), 413

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 FLASK STT API BAŞLATILIYOR")
    print("📡 Endpoint: http://localhost:5000/process")
    print("📋 Method: POST")
    print("📁 Field: audio_file")
    print("🎵 Formats: WAV, MP3, FLAC, M4A")
    print("="*50)
    
    # Development server (production'da değiştir!)
    app.run(
        host='0.0.0.0',  # Tüm IP'lerden erişim
        port=5000,
        debug=True,       # Development için
        threaded=True     # Thread support
    )