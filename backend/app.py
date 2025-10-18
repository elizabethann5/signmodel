import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from gtts import gTTS
import io
import os
from model_integration import SignLanguagePredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the sign language predictor
predictor = SignLanguagePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_frame_stream')
def handle_video_frame(data):
    try:
        # Decode base64 image
        frame_data = data.get('frame')
        if not frame_data:
            emit('error', {'message': 'No frame data received'})
            return
        
        # Remove data URL prefix if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        # Fix base64 padding if needed
        missing_padding = len(frame_data) % 4
        if missing_padding:
            frame_data += '=' * (4 - missing_padding)
        
        # Decode base64 to bytes
        frame_bytes = base64.b64decode(frame_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        
        # Decode image using OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('error', {'message': 'Failed to decode image'})
            return
        
        # Get prediction from the sign language model
        prediction_text = predictor.predict(frame)
        
        # Debug: Print frame info
        print(f"Frame received: {frame.shape if frame is not None else 'None'}")
        print(f"Prediction: '{prediction_text}'")
        
        if prediction_text and prediction_text.strip():
            # Convert text to speech
            audio_base64 = text_to_speech(prediction_text)
            
            # Send response back to frontend
            emit('translation_output', {
                'text': prediction_text,
                'audio': audio_base64,
                'timestamp': data.get('timestamp', 0)
            })
        else:
            # Send empty response to indicate no prediction
            emit('translation_output', {
                'text': '',
                'audio': None,
                'timestamp': data.get('timestamp', 0)
            })
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': f'Processing error: {str(e)}'})

def text_to_speech(text):
    """Convert text to speech and return base64 encoded audio"""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        
        return audio_base64
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return None

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)