from django.db import models

class BlogPost(models.Model):
    """Model for blog posts"""
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title

class TitleSuggestion(models.Model):
    """Model to store title suggestions"""
    content_hash = models.CharField(max_length=64, unique=True)  # Hash of the content
    original_content = models.TextField()
    suggestions = models.JSONField()  # Store list of suggested titles
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField()
    
    def __str__(self):
        return f"Suggestions for content hash: {self.content_hash[:16]}..."
