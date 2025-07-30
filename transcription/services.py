import whisper
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import tempfile
import os
import time
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class AudioTranscriptionService:
    """Service for transcribing audio files with speaker diarization"""
    
    def __init__(self):
        self.whisper_model = None
        self.diarization_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and diarization models"""
        try:
            # Load Whisper model
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
            
            # Load diarization pipeline
            # Note: You'll need to get a HuggingFace token for pyannote models
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=None  # Add your HuggingFace token here
                )
            except Exception as e:
                logger.warning(f"Diarization model not loaded: {e}")
                # Fallback: Use simple speaker detection
                self.diarization_pipeline = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file with speaker diarization
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription results with segments
        """
        start_time = time.time()
        
        logger.info(f"Starting transcription for file: {audio_file_path}")
        
        try:
            # Convert audio to format compatible with Whisper
            logger.info("Starting audio preprocessing...")
            processed_audio_path = self._preprocess_audio(audio_file_path)
            logger.info(f"Audio preprocessing complete. Processed file: {processed_audio_path}")
            
            # Transcribe with Whisper
            logger.info("Starting Whisper transcription...")
            logger.info(f"Transcribing file: {processed_audio_path}")
            logger.info(f"File exists: {os.path.exists(processed_audio_path)}")
            if os.path.exists(processed_audio_path):
                logger.info(f"File size: {os.path.getsize(processed_audio_path)} bytes")
            
            whisper_result = self.whisper_model.transcribe(processed_audio_path)
            logger.info("Whisper transcription complete")
            
            # Perform speaker diarization
            logger.info("Starting speaker diarization...")
            diarization_result = self._perform_diarization(processed_audio_path)
            logger.info("Speaker diarization complete")
            
            # Combine transcription and diarization
            segments = self._combine_transcription_and_diarization(
                whisper_result, diarization_result
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            
            # Clean up temporary files
            if processed_audio_path != audio_file_path:
                os.unlink(processed_audio_path)
            
            return {
                "segments": segments,
                "full_text": whisper_result["text"].strip(),
                "language": whisper_result.get("language", "en"),
                "duration": self._get_audio_duration(audio_file_path),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def _create_demo_transcription(self) -> Dict[str, Any]:
        """
        Create a demo transcription result to show system functionality
        when FFmpeg is not available
        """
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.2,
                    "speaker": "SPEAKER_00",
                    "text": "This is a demo transcription result."
                },
                {
                    "start": 3.5,
                    "end": 7.1,
                    "speaker": "SPEAKER_01", 
                    "text": "FFmpeg is required for actual audio processing."
                },
                {
                    "start": 7.5,
                    "end": 12.0,
                    "speaker": "SPEAKER_00",
                    "text": "Please install FFmpeg to process real audio files."
                }
            ],
            "full_text": "This is a demo transcription result. FFmpeg is required for actual audio processing. Please install FFmpeg to process real audio files.",
            "language": "en",
            "duration": 12.0,
            "processing_time": 0.5
        }
    
    def _preprocess_audio(self, audio_file_path: str) -> str:
        """
        Preprocess audio file for optimal transcription
        
        Args:
            audio_file_path: Path to the original audio file
            
        Returns:
            Path to the processed audio file
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            logger.info(f"File exists, size: {os.path.getsize(audio_file_path)} bytes")
            
            file_extension = os.path.splitext(audio_file_path)[1].lower()
            
            logger.info(f"Loading audio file with pydub: {audio_file_path}")
            
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_file_path)
            logger.info(f"Audio loaded successfully. Duration: {len(audio)}ms, Channels: {audio.channels}, Frame rate: {audio.frame_rate}")
            
            # Convert to mono and standard sample rate
            audio = audio.set_channels(1).set_frame_rate(16000)
            logger.info("Audio converted to mono 16kHz")
            
            # Export as WAV for consistent processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_wav_path = temp_file.name
            
            logger.info(f"Exporting to WAV: {temp_wav_path}")
            audio.export(temp_wav_path, format="wav")
            logger.info(f"WAV export complete. File size: {os.path.getsize(temp_wav_path)} bytes")
            
            return temp_wav_path
                
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _perform_diarization(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing speaker timeline
        """
        if self.diarization_pipeline is None:
            # Fallback: Create simple mock diarization
            return self._create_mock_diarization(audio_file_path)
        
        try:
            # Perform diarization
            diarization = self.diarization_pipeline(audio_file_path)
            
            # Convert to our format
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            return {"segments": segments}
            
        except Exception as e:
            logger.warning(f"Diarization failed: {e}. Using mock diarization.")
            return self._create_mock_diarization(audio_file_path)
    
    def _create_mock_diarization(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Create mock diarization for when the model is not available
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Mock diarization result
        """
        duration = self._get_audio_duration(audio_file_path)
        
        # Simple mock: alternate between two speakers every 10 seconds
        segments = []
        current_time = 0.0
        speaker_id = 0
        
        while current_time < duration:
            segment_duration = min(10.0, duration - current_time)
            segments.append({
                "start": current_time,
                "end": current_time + segment_duration,
                "speaker": f"SPEAKER_{speaker_id:02d}"
            })
            current_time += segment_duration
            speaker_id = 1 - speaker_id  # Alternate between 0 and 1
        
        return {"segments": segments}
    
    def _combine_transcription_and_diarization(
        self, whisper_result: Dict, diarization_result: Dict
    ) -> List[Dict[str, Any]]:
        """
        Combine Whisper transcription with speaker diarization
        
        Args:
            whisper_result: Result from Whisper transcription
            diarization_result: Result from speaker diarization
            
        Returns:
            List of segments with speaker labels and text
        """
        whisper_segments = whisper_result.get("segments", [])
        diarization_segments = diarization_result.get("segments", [])
        
        combined_segments = []
        
        for whisper_seg in whisper_segments:
            start_time = whisper_seg["start"]
            end_time = whisper_seg["end"]
            text = whisper_seg["text"].strip()
            
            # Find the speaker for this time segment
            speaker = self._find_speaker_for_time(
                start_time, end_time, diarization_segments
            )
            
            combined_segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": text
            })
        
        return combined_segments
    
    def _find_speaker_for_time(
        self, start_time: float, end_time: float, diarization_segments: List[Dict]
    ) -> str:
        """
        Find the most likely speaker for a given time segment
        
        Args:
            start_time: Start time of the segment
            end_time: End time of the segment
            diarization_segments: List of speaker segments
            
        Returns:
            Speaker label
        """
        segment_center = (start_time + end_time) / 2
        
        for dia_seg in diarization_segments:
            if dia_seg["start"] <= segment_center <= dia_seg["end"]:
                return dia_seg["speaker"]
        
        # Fallback: find the closest segment
        closest_speaker = "SPEAKER_00"
        min_distance = float('inf')
        
        for dia_seg in diarization_segments:
            seg_center = (dia_seg["start"] + dia_seg["end"]) / 2
            distance = abs(segment_center - seg_center)
            if distance < min_distance:
                min_distance = distance
                closest_speaker = dia_seg["speaker"]
        
        return closest_speaker
    
    def _get_audio_duration(self, audio_file_path: str) -> float:
        """
        Get the duration of an audio file
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            audio = AudioSegment.from_file(audio_file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except:
            return 0.0
