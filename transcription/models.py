from django.db import models

class AudioFile(models.Model):
    """Model to store uploaded audio files"""
    file = models.FileField(upload_to='audio/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.IntegerField()
    duration = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Audio file {self.id} - {self.file.name}"

class TranscriptionResult(models.Model):
    """Model to store transcription results"""
    audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE)
    full_text = models.TextField()
    language = models.CharField(max_length=10, default='en')
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField()
    
    def __str__(self):
        return f"Transcription for {self.audio_file.file.name}"

class TranscriptionSegment(models.Model):
    """Model to store individual transcription segments with speaker info"""
    transcription = models.ForeignKey(TranscriptionResult, on_delete=models.CASCADE, related_name='segments')
    start_time = models.FloatField()
    end_time = models.FloatField()
    speaker = models.CharField(max_length=50)
    text = models.TextField()
    
    class Meta:
        ordering = ['start_time']
    
    def __str__(self):
        return f"{self.speaker}: {self.text[:50]}..."
