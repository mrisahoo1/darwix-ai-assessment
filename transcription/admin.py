from django.contrib import admin
from .models import AudioFile, TranscriptionResult, TranscriptionSegment

@admin.register(AudioFile)
class AudioFileAdmin(admin.ModelAdmin):
    list_display = ['id', 'file', 'uploaded_at', 'file_size', 'duration']
    list_filter = ['uploaded_at']
    search_fields = ['file']
    readonly_fields = ['uploaded_at']

@admin.register(TranscriptionResult)
class TranscriptionResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'audio_file', 'language', 'created_at', 'processing_time']
    list_filter = ['language', 'created_at']
    search_fields = ['full_text']
    readonly_fields = ['created_at']

@admin.register(TranscriptionSegment)
class TranscriptionSegmentAdmin(admin.ModelAdmin):
    list_display = ['id', 'transcription', 'speaker', 'start_time', 'end_time', 'text']
    list_filter = ['speaker']
    search_fields = ['text', 'speaker']
