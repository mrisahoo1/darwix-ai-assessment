# Darwix AI Assessment Project

**Implementation of Audio Transcription with Speaker Diarization and AI-Powered Blog Title Generation**

![Python](https://i## 📊 **Code Quality Features**g.shields.io/badge/Python-3.9+-blue?logo=python)
![Django](https://img.shields.io/badge/Django-4.2+-green?logo=django)
![AI](https://img.shields.io/badge/AI-Whisper%20%7C%20NLP-orange)
![Status](https://img.shields.io/badge/Status-Assessment%20Ready-success)

## � **Assessment Requirements Fulfilled**

### **Feature 1: Audio Transcription with Diarization**
- **Audio Transcription**: Implemented using OpenAI Whisper for high-accuracy speech recognition
- **Speaker Diarization**: "Who spoke when" identification using pyannote.audio
- **Multi-format Support**: WAV, MP3, MP4, M4A, FLAC, AAC
- **Structured JSON Output**: Timestamped segments with speaker labels
- **Multilingual Support**: 39 languages auto-detected (bonus feature)

### **Feature 2: AI Title Suggestions for Blog Posts**
- **NLP Integration**: Content-aware title generation using TF-IDF and NLTK - [Better results can be achieved with BERT/BART]
- **Django Integration**: RESTful API endpoint with proper error handling
- **3 Title Suggestions**: Multiple algorithmic approaches for variety
- **JSON Response Format**: Structured output with metadata
- **Content Analysis**: Intelligent keyword extraction and pattern recognition

## **Quick Setup & Testing**

### **1. Installation**
```bash
# Clone repository
git clone https://github.com/mrisahoo1/darwix-ai-assessment.git
cd darwix-ai-assessment

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Initialize database
python manage.py migrate

# Start server
python manage.py runserver
```

### **2. Test the Application**
- **Web Interface**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/swagger/
## 🔧 **API Endpoints**

### **Feature 1: Audio Transcription with Diarization**

**Endpoint**: `POST /api/transcribe/`

**Request**:
```bash
curl -X POST \
  http://127.0.0.1:8000/api/transcribe/ \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio_file=@Recording.m4a'
```

**Response**:
```json
{
  "status": "success",
  "full_transcription": "Hello, I am taking a test object for Darwix AI project. Thank you.",
  "diarization": [
    {
      "start": 0.0,
      "end": 3.2,
      "speaker": "SPEAKER_00",
      "text": "Hello, I am taking a test object for Darwix AI project."
    },
    {
      "start": 3.2,
      "end": 4.1,
      "speaker": "SPEAKER_01",
      "text": "Thank you."
    }
  ],
  "metadata": {
    "language": "en",
    "duration": 4.83,
    "processing_time": 2.15
  }
}
```

### **Feature 2: Blog Post Title Suggestions**

**Endpoint**: `POST /api/suggest-titles/`

**Request**:
```bash
curl -X POST \
  http://127.0.0.1:8000/api/suggest-titles/ \
  -H 'Content-Type: application/json' \
  -d '{
    "content": "Artificial intelligence is revolutionizing industries. Machine learning algorithms are enabling automation of complex tasks that previously required human intelligence."
  }'
```

**Response**:
```json
{
  "status": "success",
  "suggestions": [
    "How AI is Revolutionizing Industries Through Automation",
    "Machine Learning: Transforming Complex Task Automation",
    "The Future of Artificial Intelligence in Business"
  ],
  "content_length": 156,
  "processing_time": 1.45,
  "cached": false
}
```

## 🏗️ **Technical Implementation**

### **Technologies Used**
- **Django 4.2.7**: Web framework and REST API
- **OpenAI Whisper**: Speech recognition and transcription
- **pyannote.audio**: Speaker diarization pipeline
- **FFmpeg**: Audio preprocessing
- **TF-IDF + NLTK**: Natural language processing for title generation
- **Django REST Framework**: API endpoints and serialization

### **Project Structure**
```
darwix_assessment/
├── darwix_project/         # Django configuration
│   ├── settings.py        # Project settings
│   ├── urls.py           # Main URL routing
│   └── wsgi.py           # WSGI application
├── transcription/          # Feature 1: Audio transcription
│   ├── views.py           # API endpoint for transcription
│   ├── services.py        # Whisper & diarization logic
│   ├── models.py          # Data models
│   └── urls.py           # App routing
├── blog/                  # Feature 2: Title suggestions
│   ├── views.py           # API endpoint for title generation
│   ├── services.py        # NLP processing logic
│   ├── models.py          # Blog models
│   └── urls.py           # App routing
├── frontend/              # Web interface for testing
│   ├── views.py           # Frontend views
│   ├── templates/         # HTML templates
│   └── static/           # CSS, JS assets
├── media/                 # Sample audio files
│   └── audio/            # Audio file storage
│       └── Recording.m4a # Sample test file
├── ffmpeg/               # Local FFmpeg installation
├── requirements.txt      # Python dependencies
├── manage.py            # Django management
├── .env                 # Environment configuration
└── README.md           # This documentation
```

## 🧪 **Testing Instructions**

### **Using Web Interface**
1. Navigate to http://127.0.0.1:8000
2. **Audio Transcription**: Upload `Recording.m4a` or any audio file
3. **Title Generation**: Enter blog content and generate titles

### **Using API Directly**
1. **Test Audio Transcription**: Use the curl command above with the provided `Recording.m4a`
2. **Test Title Generation**: Send POST request with blog content as shown above

### **Sample Test Data**
- **Audio File**: `Recording.m4a` (included in project)
- **Blog Content**: Any text content for title generation testing

## **Code Quality Features**

### **Modularity**
- Separate Django apps for each feature
- Service layer pattern for business logic
- Clean separation of concerns

### **Maintainability**
- Comprehensive error handling
- Logging for debugging
- Structured JSON responses
- Type hints and documentation

### **Robustness**
- Input validation
- File format support
- Graceful error handling
- Performance monitoring

## **Ready for Assessment**

This project fully implements both required features:

1. **✅ Audio Transcription with Diarization**: Working with real audio files, multilingual support
2. **✅ AI Title Suggestions**: Content-aware NLP generating 3 relevant titles

**Live Demo**: http://127.0.0.1:8000

**GitHub**: https://github.com/mrisahoo1/darwix-ai-assessment

The implementation demonstrates effective AI/NLP integration into Django with clean, maintainable code and comprehensive documentation.

## Technology Stack

- **Backend:** Django 4.2, Django REST Framework
- **AI/ML:** OpenAI Whisper, pyannote.audio, Transformers (BERT)
- **Audio Processing:** pydub, torch, torchaudio
- **NLP:** NLTK, scikit-learn
- **Documentation:** drf-yasg (Swagger/OpenAPI)

## License

This project is developed as part of the Darwix AI technical assessment.

## Support

For questions or issues, please contact the development team.
