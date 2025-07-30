from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging

from .services import TitleSuggestionService
from .models import TitleSuggestion

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class TitleSuggestionView(APIView):
    """
    API endpoint for generating AI-powered blog title suggestions
    """
    
    @swagger_auto_schema(
        operation_description="Generate 3 AI-powered title suggestions for blog content",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['content'],
            properties={
                'content': openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="Blog post content"
                )
            }
        ),
        responses={
            200: openapi.Response(
                description="Title suggestions generated successfully",
                examples={
                    "application/json": {
                        "status": "success",
                        "suggestions": [
                            "How AI is Revolutionizing Our Daily Lives and Work",
                            "The Rise of Machine Learning: Transforming Complex Tasks",
                            "Artificial Intelligence: The Future of Automated Intelligence"
                        ],
                        "content_length": 234,
                        "processing_time": 0.45
                    }
                }
            ),
            400: openapi.Response(description="Bad request - no content provided"),
            500: openapi.Response(description="Internal server error")
        }
    )
    def post(self, request):
        """
        Generate title suggestions for blog content
        """
        try:
            # Get content from request
            content = request.data.get('content', '').strip()
            
            if not content:
                return Response(
                    {
                        "status": "error",
                        "message": "No content provided. Please include blog content in the request."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            if len(content) < 20:
                return Response(
                    {
                        "status": "error",
                        "message": "Content too short. Please provide at least 20 characters of content."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Initialize title suggestion service
            title_service = TitleSuggestionService()
            
            # Generate fresh suggestions (no caching for now to ensure uniqueness)
            result = title_service.generate_title_suggestions(content)
            
            # Cache the results for future use
            content_hash = title_service.get_content_hash(content)
            try:
                TitleSuggestion.objects.create(
                    content_hash=content_hash,
                    original_content=content[:1000],  # Store first 1000 chars
                    suggestions=result['suggestions'],
                    processing_time=result['processing_time']
                )
            except Exception as e:
                logger.warning(f"Failed to cache title suggestions: {e}")
            
            # Prepare response  
            response_data = {
                "status": "success",
                "suggestions": result['suggestions'],
                "content_length": result['content_length'],
                "processing_time": round(result['processing_time'], 2),
                "cached": False
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error generating title suggestions: {str(e)}")
            return Response(
                {
                    "status": "error",
                    "message": "An error occurred while generating title suggestions. Please try again."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
