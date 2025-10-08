import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
from flask import Flask, request, jsonify
import tempfile
import os
import time
import re

# Initialize Flask app
app = Flask(__name__)

print("Loading Enhanced Wav2Vec2 model...")
# Using a larger, more accurate model
model_name = "facebook/wav2vec2-large-960h-lv60-self"  # Much more accurate model

try:
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    print("✅ Enhanced model loaded successfully!")
    print("📊 This model was trained on 960 hours of speech data")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("🔄 Falling back to base model...")
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

def clean_transcription(text):
    """Clean and format the transcription for better readability"""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Capitalize first letter of each sentence
    sentences = text.split('. ')
    sentences = [sentence.strip().capitalize() for sentence in sentences if sentence.strip()]
    text = '. '.join(sentences)
    
    # Ensure proper punctuation at the end
    if text and not text[-1] in ['.', '!', '?']:
        text += '.'
    
    # Fix common transcription errors - using proper string escaping
    common_fixes = {
        r'\bi\b': 'I',
        r'\bim\b': "I'm",
        r'\bive\b': "I've",
        r'\byoure\b': "you're",
        r'\bwere\b': "we're",
        r'\btheres\b': "there's",
        r'\bdont\b': "don't",
        r'\bwont\b': "won't",
        r'\bcant\b': "can't",
        r'\bshouldnt\b': "shouldn't",
        r'\bcouldnt\b': "couldn't",
        r'\bwouldnt\b': "wouldn't",
        r'\bisnt\b': "isn't",
        r'\baren\'t\b': "aren't",  # Fixed this line
        r'\bwasnt\b': "wasn't",
        r'\bwerent\b': "weren't",
        r'\bhavent\b': "haven't",
        r'\bhasnt\b': "hasn't",
        r'\bhadnt\b': "hadn't",
        r'\bdoesnt\b': "doesn't",
        r'\bdidnt\b': "didn't",
    }
    
    for pattern, replacement in common_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text.strip()

def transcribe_audio(audio_data, sample_rate=16000):
    """Transcribe audio data to text using Wav2Vec2 model with better processing"""
    try:
        # Resample if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Normalize audio for better recognition
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Process the audio
        input_values = processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        ).input_values
        
        # Get model predictions with attention to detail
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Get the predicted token ids
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode the audio to text
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Clean and format the transcription
        cleaned_transcription = clean_transcription(transcription)
        
        return cleaned_transcription
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Sorry, there was an error processing the audio."

def safe_delete_file(file_path, max_retries=5, delay=0.1):
    """Safely delete a file with retries to handle Windows file locking"""
    for i in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                return True
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"⚠️ Warning: Could not delete temporary file: {file_path}")
                return False
    return False

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Professional Audio Transcription</title>
        <style>
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
            }
            
            body { 
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; 
                padding: 20px;
                color: #333;
            }
            
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 12px; 
                box-shadow: 0 15px 35px rgba(0,0,0,0.1); 
                overflow: hidden; 
            }
            
            .header { 
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                color: white; 
                padding: 50px 40px; 
                text-align: center; 
            }
            
            .header h1 { 
                font-size: 2.8em; 
                margin-bottom: 15px;
                font-weight: 300;
                letter-spacing: -0.5px;
            }
            
            .header p {
                font-size: 1.3em;
                opacity: 0.95;
                font-weight: 300;
            }
            
            .model-info {
                background: rgba(255,255,255,0.1);
                padding: 10px 20px;
                border-radius: 20px;
                margin-top: 15px;
                display: inline-block;
                font-size: 0.9em;
            }
            
            .content { 
                padding: 50px; 
            }
            
            .upload-area { 
                border: 2px dashed #b8c2cc; 
                border-radius: 10px; 
                padding: 50px 40px; 
                text-align: center; 
                margin-bottom: 30px; 
                transition: all 0.3s ease; 
                cursor: pointer;
                background: #fafbfc;
            }
            
            .upload-area:hover, .upload-area.dragover {
                background: #f0f4f8;
                border-color: #4facfe;
            }
            
            .upload-icon { 
                font-size: 3.5em; 
                color: #4facfe; 
                margin-bottom: 25px;
                opacity: 0.8;
            }
            
            .upload-area h3 {
                color: #2d3748;
                margin-bottom: 12px;
                font-size: 1.6em;
                font-weight: 500;
            }
            
            .upload-area p {
                color: #718096;
                margin-bottom: 8px;
                font-size: 1.1em;
            }
            
            .file-types {
                color: #a0aec0;
                font-size: 0.95em;
                margin-top: 15px;
            }
            
            .btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                border: none; 
                padding: 14px 35px; 
                border-radius: 8px; 
                font-size: 1.1em; 
                cursor: pointer; 
                transition: all 0.3s ease; 
                margin: 15px 0;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            
            .btn:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn:active {
                transform: translateY(0);
            }
            
            .btn.secondary {
                background: #718096;
            }
            
            .btn.secondary:hover {
                background: #4a5568;
                box-shadow: 0 5px 15px rgba(113, 128, 150, 0.4);
            }
            
            .btn.success {
                background: #48bb78;
            }
            
            .btn.success:hover {
                background: #38a169;
                box-shadow: 0 5px 15px rgba(72, 187, 120, 0.4);
            }
            
            #audioFile { 
                display: none; 
            }
            
            .loading { 
                display: none; 
                text-align: center; 
                padding: 50px 40px; 
            }
            
            .spinner { 
                border: 3px solid #f3f3f3; 
                border-top: 3px solid #4facfe; 
                border-radius: 50%; 
                width: 50px; 
                height: 50px; 
                animation: spin 1s linear infinite; 
                margin: 0 auto 25px; 
            }
            
            @keyframes spin { 
                0% { transform: rotate(0deg); } 
                100% { transform: rotate(360deg); } 
            }
            
            .loading p {
                color: #718096;
                font-size: 1.1em;
            }
            
            .result-area { 
                display: none; 
                background: #f7fafc; 
                border-radius: 10px; 
                padding: 40px; 
                margin-top: 20px; 
                border: 1px solid #e2e8f0;
            }
            
            .result-area h3 {
                color: #2d3748;
                margin-bottom: 25px;
                font-size: 1.5em;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .audio-player { 
                width: 100%; 
                margin: 25px 0; 
                border-radius: 8px;
                background: #f7fafc;
            }
            
            .transcription { 
                background: white; 
                padding: 30px; 
                border-radius: 8px; 
                border-left: 4px solid #4facfe; 
                font-size: 1.15em; 
                line-height: 1.7; 
                margin: 20px 0; 
                min-height: 150px;
                color: #2d3748;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            .file-details { 
                background: #edf2f7; 
                padding: 20px; 
                border-radius: 8px; 
                margin: 20px 0; 
                font-size: 0.95em;
                color: #4a5568;
                border-left: 4px solid #cbd5e0;
            }
            
            .file-details strong {
                color: #2d3748;
            }
            
            .confidence {
                background: #e6fffa;
                padding: 15px 20px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #4fd1c7;
                font-size: 0.9em;
            }
            
            .action-buttons {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-top: 25px;
                flex-wrap: wrap;
            }
            
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                    border-radius: 8px;
                }
                
                .header {
                    padding: 40px 20px;
                }
                
                .header h1 {
                    font-size: 2.2em;
                }
                
                .content {
                    padding: 30px 20px;
                }
                
                .upload-area {
                    padding: 30px 20px;
                }
                
                .action-buttons {
                    flex-direction: column;
                }
                
                .btn {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Professional Audio Transcription</h1>
                <p>High-accuracy speech-to-text conversion</p>
                <div class="model-info">Powered by Wav2Vec2 Large Model</div>
            </div>
            
            <div class="content">
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon"></div>
                    <h3>Upload Audio for Transcription</h3>
                    <p>For best results, use clear audio with minimal background noise</p>
                    <p class="file-types">Supported formats: WAV, MP3, FLAC, M4A, OPUS, OGG</p>
                    <p class="file-types">Maximum file size: 25MB • Recommended: 16kHz sample rate</p>
                    <input type="file" id="audioFile" accept="audio/*">
                    <button class="btn" onclick="document.getElementById('audioFile').click()">
                        Select Audio File
                    </button>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing your audio with high-accuracy model...</p>
                </div>
                
                <div class="result-area" id="resultArea">
                    <h3>Transcription Result</h3>
                    <audio controls class="audio-player" id="audioPlayer"></audio>
                    <div class="confidence">
                        <strong>Enhanced Accuracy:</strong> Using large model trained on 960 hours of speech data
                    </div>
                    <div class="transcription" id="transcriptionText"></div>
                    <div class="file-details" id="fileDetails"></div>
                    <div class="action-buttons">
                        <button class="btn secondary" onclick="resetForm()">Transcribe Another File</button>
                        <button class="btn success" onclick="copyTranscription()">Copy Text</button>
                        <button class="btn" onclick="downloadTranscription()">Download Text</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const audioFileInput = document.getElementById('audioFile');
            const loading = document.getElementById('loading');
            const resultArea = document.getElementById('resultArea');
            const transcriptionText = document.getElementById('transcriptionText');
            const audioPlayer = document.getElementById('audioPlayer');
            const fileDetails = document.getElementById('fileDetails');
            
            // Enhanced drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                if (e.dataTransfer.files.length) {
                    audioFileInput.files = e.dataTransfer.files;
                    handleFileUpload();
                }
            });
            
            audioFileInput.addEventListener('change', handleFileUpload);
            
            function handleFileUpload() {
                const file = audioFileInput.files[0];
                if (!file) return;
                
                // Enhanced file validation
                const validTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/x-m4a', 'audio/ogg', 'audio/opus'];
                const fileType = file.type.toLowerCase();
                
                if (!validTypes.some(type => fileType.includes(type.replace('audio/', ''))) && 
                    !file.name.toLowerCase().endsWith('.opus')) {
                    alert('Please upload a valid audio file (WAV, MP3, FLAC, M4A, OPUS, OGG)');
                    return;
                }
                
                if (file.size > 25 * 1024 * 1024) {
                    alert('File size must be less than 25MB');
                    return;
                }
                
                uploadArea.style.display = 'none';
                loading.style.display = 'block';
                
                const formData = new FormData();
                formData.append('audio', file);
                
                fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (data.status === 'success') {
                        transcriptionText.textContent = data.transcription;
                        audioPlayer.src = URL.createObjectURL(file);
                        fileDetails.innerHTML = `
                            <strong>File Name:</strong> ${file.name}<br>
                            <strong>Duration:</strong> ${data.audio_duration}<br>
                            <strong>File Size:</strong> ${(file.size / (1024 * 1024)).toFixed(2)} MB<br>
                            <strong>Status:</strong> High-accuracy transcription complete
                        `;
                        resultArea.style.display = 'block';
                        
                        // Smooth scroll to results
                        resultArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    } else {
                        throw new Error(data.error || 'Transcription failed');
                    }
                })
                .catch(error => {
                    loading.style.display = 'none';
                    uploadArea.style.display = 'block';
                    alert('Error: ' + error.message);
                    console.error('Upload error:', error);
                });
            }
            
            function resetForm() {
                audioFileInput.value = '';
                resultArea.style.display = 'none';
                uploadArea.style.display = 'block';
                uploadArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            function copyTranscription() {
                const text = transcriptionText.textContent;
                navigator.clipboard.writeText(text).then(() => {
                    alert('Transcription copied to clipboard!');
                }).catch(() => {
                    alert('Failed to copy text. Please select and copy manually.');
                });
            }
            
            function downloadTranscription() {
                const text = transcriptionText.textContent;
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'transcription.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        </script>
    </body>
    </html>
    '''

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    temp_file_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.opus', '.ogg'}
        file_ext = os.path.splitext(audio_file.filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not supported. Use: WAV, MP3, FLAC, M4A, OPUS, OGG'}), 400

        # Create temporary file with unique name
        temp_file_path = tempfile.mktemp(suffix=file_ext)
        audio_file.save(temp_file_path)
        
        # Load and process the audio file
        try:
            audio_data, sample_rate = librosa.load(temp_file_path, sr=16000)
        except Exception as e:
            return jsonify({'error': f'Error loading audio file: {str(e)}'}), 400
        
        # Transcribe the audio with enhanced model
        transcription = transcribe_audio(audio_data, sample_rate)
        
        return jsonify({
            'status': 'success',
            'transcription': transcription,
            'audio_duration': f"{len(audio_data)/sample_rate:.2f} seconds",
            'filename': audio_file.filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Always try to clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            safe_delete_file(temp_file_path)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Transcription Server...")
    print("📱 Open http://127.0.0.1:5000 in your browser")
    print("🎯 Using high-accuracy Wav2Vec2 Large model")
    app.run(debug=True, host='127.0.0.1', port=5000)