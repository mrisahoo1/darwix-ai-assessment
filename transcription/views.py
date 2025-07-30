from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import os
import tempfile
import logging

from .services import AudioTranscriptionService
from .models import AudioFile, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class TranscriptionView(APIView):
    """
    API endpoint for audio transcription with speaker diarization
    """
    parser_classes = (MultiPartParser, FormParser)
    
    @swagger_auto_schema(
        operation_description="Transcribe audio file with speaker diarization",
        manual_parameters=[
            openapi.Parameter(
                'audio_file',
                openapi.IN_FORM,
                description="Audio file to transcribe (WAV, MP3, MP4, etc.)",
                type=openapi.TYPE_FILE,
                required=True
            )
        ],
        responses={
            200: openapi.Response(
                description="Transcription successful",
                examples={
                    "application/json": {
                        "status": "success",
                        "transcription": {
                            "segments": [
                                {
                                    "start": 0.0,
                                    "end": 5.2,
                                    "speaker": "SPEAKER_00",
                                    "text": "Hello, how are you today?"
                                }
                            ],
                            "full_text": "Hello, how are you today?",
                            "language": "en",
                            "duration": 8.1
                        }
                    }
                }
            ),
            400: openapi.Response(description="Bad request - no audio file provided"),
            500: openapi.Response(description="Internal server error")
        }
    )
    def post(self, request):
        """
        Handle audio file upload and transcription
        """
        try:
            # Check if audio file is provided
            if 'audio_file' not in request.FILES:
                return Response(
                    {
                        "status": "error",
                        "message": "No audio file provided. Please upload an audio file."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            audio_file = request.FILES['audio_file']
            
            # Validate file type
            allowed_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.aac']
            file_extension = os.path.splitext(audio_file.name)[1].lower()
            
            if file_extension not in allowed_extensions:
                return Response(
                    {
                        "status": "error",
                        "message": f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Save the uploaded file
            audio_file_obj = AudioFile.objects.create(
                file=audio_file,
                file_size=audio_file.size
            )
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            # Verify file was created and exists
            if not os.path.exists(temp_file_path):
                return Response(
                    {
                        "status": "error",
                        "message": f"Failed to create temporary file: {temp_file_path}"
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            logger.info(f"Created temporary file: {temp_file_path}, size: {os.path.getsize(temp_file_path)}")
            
            try:
                # Initialize transcription service
                transcription_service = AudioTranscriptionService()
                
                # Perform transcription
                result = transcription_service.transcribe_audio(temp_file_path)
                
                # Update audio file duration
                audio_file_obj.duration = result['duration']
                audio_file_obj.save()
                
                # Save transcription result
                transcription_obj = TranscriptionResult.objects.create(
                    audio_file=audio_file_obj,
                    full_text=result['full_text'],
                    language=result['language'],
                    processing_time=result['processing_time']
                )
                
                # Save segments
                for segment_data in result['segments']:
                    TranscriptionSegment.objects.create(
                        transcription=transcription_obj,
                        start_time=segment_data['start'],
                        end_time=segment_data['end'],
                        speaker=segment_data['speaker'],
                        text=segment_data['text']
                    )
                
                # Prepare response
                response_data = {
                    "status": "success",
                    "full_transcription": result['full_text'],
                    "diarization": result['segments'],
                    "metadata": {
                        "language": result['language'],
                        "duration": round(result['duration'], 2),
                        "processing_time": round(result.get('processing_time', 0), 2)
                    }
                }
                
                return Response(response_data, status=status.HTTP_200_OK)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            # For debugging, return the actual error message
            return Response(
                {
                    "status": "error",
                    "message": f"Transcription error: {str(e)}"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
