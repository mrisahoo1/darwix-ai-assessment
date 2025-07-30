from django.contrib import admin
from .models import BlogPost, TitleSuggestion

@admin.register(BlogPost)
class BlogPostAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'created_at', 'updated_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['title', 'content']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(TitleSuggestion)
class TitleSuggestionAdmin(admin.ModelAdmin):
    list_display = ['id', 'content_hash', 'created_at', 'processing_time']
    list_filter = ['created_at']
    search_fields = ['content_hash', 'original_content']
    readonly_fields = ['created_at']
