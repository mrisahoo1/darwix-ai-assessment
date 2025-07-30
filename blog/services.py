from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import re
import hashlib
import time
import logging
from typing import List, Dict, Any
import torch
import requests
import json

logger = logging.getLogger(__name__)

class TitleSuggestionService:
    """Service for generating AI-powered blog title suggestions"""
    
    def __init__(self):
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        self.tfidf_vectorizer = None
        self._initialize_models()
        self._download_nltk_data()
    
    def _initialize_models(self):
        """Initialize lightweight NLP models for title generation"""
        try:
            # Use smaller, lighter models to avoid memory issues
            # Try to use lightweight summarization pipeline
            try:
                # Use a smaller model for summarization
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-6-6",  # Much smaller than bart-large-cnn
                    tokenizer="sshleifer/distilbart-cnn-6-6",
                    device=-1  # Force CPU usage to avoid GPU memory issues
                )
                logger.info("Lightweight summarization model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load summarization model: {e}")
                self.summarizer = None
            
            # Use smaller BERT model
            try:
                model_name = "distilbert-base-uncased"  # Smaller than bert-base-uncased
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                logger.info("DistilBERT model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BERT model: {e}")
                self.tokenizer = None
                self.model = None
            
            # Initialize TF-IDF vectorizer (lightweight, no model download needed)
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,  # Reduced from 1000
                stop_words='english',
                ngram_range=(1, 2),  # Reduced from (1, 3)
                min_df=1,
                max_df=0.8
            )
            
            logger.info("Title suggestion service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Complete fallback - no models loaded
            self.summarizer = None
            self.tokenizer = None
            self.model = None
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            logger.warning("NLTK data download failed")
    
    def generate_title_suggestions(self, content: str) -> Dict[str, Any]:
        """
        Generate 3 unique title suggestions for blog content (3-15 words each)
        
        Args:
            content: Blog post content
            
        Returns:
            Dictionary containing suggestions and metadata
        """
        start_time = time.time()
        
        try:
            # Clean and preprocess content
            cleaned_content = self._preprocess_content(content)
            
            if len(cleaned_content.strip()) < 20:
                return self._generate_fallback_titles(content, start_time)
            
            # Generate suggestions using different approaches (no caching influences)
            all_suggestions = []
            
            # Approach 1: Content-based analysis (primary method)
            content_titles = self._generate_content_based_titles(cleaned_content)
            all_suggestions.extend(content_titles)
            
            # Approach 2: Keyword-based titles
            keyword_titles = self._generate_keyword_titles(cleaned_content)
            all_suggestions.extend(keyword_titles)
            
            # Approach 3: Question-based titles
            question_titles = self._generate_question_titles(cleaned_content)
            all_suggestions.extend(question_titles)
            
            # Approach 4: Structure-based titles
            structure_titles = self._generate_structure_based_titles(cleaned_content)
            all_suggestions.extend(structure_titles)
            
            # Approach 5: Extractive summarization (if model available)
            if self.summarizer:
                try:
                    extractive_titles = self._generate_extractive_titles(cleaned_content)
                    all_suggestions.extend(extractive_titles)
                except Exception as e:
                    logger.warning(f"Extractive summarization failed: {e}")
            
            # Select best 3 unique suggestions based purely on content
            final_suggestions = self._select_best_suggestions(all_suggestions, content)
            
            # Ensure we have exactly 3 unique titles
            if len(final_suggestions) < 3:
                additional_titles = self._generate_additional_titles(cleaned_content, final_suggestions)
                final_suggestions.extend(additional_titles)
            
            processing_time = time.time() - start_time
            
            return {
                "suggestions": final_suggestions[:3],  # Ensure exactly 3
                "content_length": len(content),
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error generating title suggestions: {e}")
            return self._generate_fallback_titles(content, start_time)
    
    def _generate_fallback_titles(self, content: str, start_time: float) -> Dict[str, Any]:
        """Generate basic titles when models fail"""
        # Extract key information from content
        words = content.lower().split()
        
        # Simple keyword extraction
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        key_words = [w for w in words if len(w) > 3 and w not in common_words][:5]
        
        suggestions = []
        if key_words:
            main_topic = key_words[0].title()
            suggestions = [
                f"Understanding {main_topic}",
                f"A Guide to {main_topic}",
                f"Everything About {main_topic}"
            ]
        else:
            suggestions = [
                "New Insights and Ideas",
                "Important Information",
                "Key Points to Consider"
            ]
        
        return {
            "suggestions": suggestions,
            "content_length": len(content),
            "processing_time": time.time() - start_time
        }
    
    def _generate_content_based_titles(self, content: str) -> List[str]:
        """
        Generate creative titles based on deep content analysis (3-15 words each)
        
        Args:
            content: Blog content
            
        Returns:
            List of creative title suggestions
        """
        try:
            sentences = sent_tokenize(content)
            if not sentences:
                return []
            
            # Extract entities and key phrases
            key_phrases = self._extract_key_phrases(content)
            if not key_phrases:
                return []
            
            main_topic = key_phrases[0]
            secondary_topic = key_phrases[1] if len(key_phrases) > 1 else None
            content_lower = content.lower()
            
            titles = []
            
            # Analyze content sentiment and context
            is_technical = any(word in content_lower for word in ['algorithm', 'technology', 'system', 'process', 'method', 'technique'])
            is_business = any(word in content_lower for word in ['business', 'company', 'market', 'revenue', 'profit', 'strategy'])
            is_future_oriented = any(word in content_lower for word in ['future', 'next', 'coming', 'ahead', 'tomorrow', 'emerging'])
            is_educational = any(word in content_lower for word in ['learn', 'understand', 'knowledge', 'skill', 'education'])
            
            # Pattern 1: Problem-solution focus
            if any(word in content_lower for word in ['problem', 'solution', 'challenge', 'issue', 'fix', 'solve']):
                if is_business:
                    titles.append(f"Solving {main_topic} Challenges in Modern Business")
                elif is_technical:
                    titles.append(f"The {main_topic} Problem: Solutions That Work")
                else:
                    titles.append(f"Overcoming {main_topic} Challenges: A Practical Guide")
            
            # Pattern 2: Transformation/change focus  
            if any(word in content_lower for word in ['transform', 'change', 'revolutioniz', 'improve', 'innovat', 'evolv']):
                if is_future_oriented:
                    titles.append(f"How {main_topic} Is Reshaping Our Future")
                elif is_business:
                    titles.append(f"The {main_topic} Transformation: Business Impact")
                else:
                    titles.append(f"{main_topic}: Driving Real Change and Innovation")
            
            # Pattern 3: Benefits/advantages focus
            if any(word in content_lower for word in ['benefit', 'advantage', 'value', 'positive', 'gain']):
                titles.append(f"The Hidden Benefits of {main_topic}")
            
            # Pattern 4: Comparison/contrast
            if secondary_topic and any(word in content_lower for word in ['vs', 'versus', 'compared', 'difference', 'between']):
                titles.append(f"{main_topic} vs {secondary_topic}: Which Wins?")
            
            # Pattern 5: How-to/instructional
            if is_educational or any(word in content_lower for word in ['step', 'guide', 'tutorial', 'how']):
                titles.append(f"Mastering {main_topic}: The Complete Roadmap")
            
            # Pattern 6: Industry secrets/expert insights
            if any(word in content_lower for word in ['expert', 'professional', 'industry', 'insider', 'secret']):
                titles.append(f"Industry Experts Reveal {main_topic} Secrets")
            
            # Pattern 7: Trend/future analysis
            if is_future_oriented:
                titles.append(f"The Future of {main_topic}: Trends to Watch")
            
            # Pattern 8: Impact/importance
            if any(word in content_lower for word in ['important', 'crucial', 'essential', 'vital', 'critical']):
                titles.append(f"Why {main_topic} Matters More Than Ever")
            
            # Filter and format titles
            formatted_titles = []
            for title in titles:
                formatted_title = self._format_as_title(title)
                if formatted_title and 3 <= len(formatted_title.split()) <= 15:
                    formatted_titles.append(formatted_title)
            
            return formatted_titles[:4]  # Return up to 4 for variety
            
        except Exception as e:
            logger.warning(f"Content-based title generation failed: {e}")
            return []
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content"""
        try:
            # Use TF-IDF if available
            if self.tfidf_vectorizer:
                try:
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    tfidf_scores = tfidf_matrix.toarray()[0]
                    
                    # Get top phrases
                    top_indices = tfidf_scores.argsort()[-5:][::-1]
                    key_phrases = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0.1]
                    return [phrase.title() for phrase in key_phrases]
                except:
                    pass
            
            # Fallback: simple frequency analysis
            words = word_tokenize(content.lower())
            
            # Get common stopwords
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # Count meaningful words
            word_freq = {}
            for word in words:
                if (word.isalpha() and 
                    len(word) > 3 and 
                    word not in stop_words):
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top words
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            return [word.title() for word, freq in top_words]
            
        except Exception as e:
            logger.warning(f"Key phrase extraction failed: {e}")
            return []
    
    def _extract_core_words(self, sentence: str) -> List[str]:
        """Extract core meaningful words from a sentence"""
        try:
            words = word_tokenize(sentence.lower())
            
            # Simple POS tagging fallback
            try:
                tagged = pos_tag(words)
                # Focus on nouns and adjectives
                core_words = [word for word, pos in tagged if pos.startswith(('NN', 'JJ')) and len(word) > 3]
            except:
                # Fallback without POS tagging
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                core_words = [word for word in words if word.isalpha() and len(word) > 3 and word not in stop_words]
            
            return core_words[:3]
            
        except Exception as e:
            logger.warning(f"Core word extraction failed: {e}")
            return []
    
    def _preprocess_content(self, content: str) -> str:
        """
        Clean and preprocess the content
        
        Args:
            content: Raw blog content
            
        Returns:
            Cleaned content
        """
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters but keep punctuation
        content = re.sub(r'[^\w\s.,!?;:-]', '', content)
        
        return content.strip()
    
    def _generate_extractive_titles(self, content: str) -> List[str]:
        """
        Generate titles using lightweight extractive summarization
        
        Args:
            content: Blog content
            
        Returns:
            List of title suggestions
        """
        if not self.summarizer:
            return []
        
        try:
            # Limit content length for the model (smaller limit for lighter model)
            max_length = 512  # Reduced from 1024
            if len(content) > max_length:
                # Take first part of content for title generation
                content = content[:max_length]
            
            # Generate summary with conservative parameters
            summary = self.summarizer(
                content,
                max_length=30,  # Shorter summaries
                min_length=5,
                do_sample=False,
                num_beams=2  # Reduced beam search
            )
            
            summary_text = summary[0]['summary_text']
            
            # Convert summary to title format
            title = self._format_as_title(summary_text)
            
            return [title] if title and len(title) > 5 else []
            
        except Exception as e:
            logger.warning(f"Extractive title generation failed: {e}")
            return []
    
    def _generate_keyword_titles(self, content: str) -> List[str]:
        """
        Generate creative titles based on important keywords (3-15 words)
        
        Args:
            content: Blog content
            
        Returns:
            List of creative keyword-based titles
        """
        try:
            # Extract sentences
            sentences = sent_tokenize(content)
            if not sentences:
                return []
            
            # Get key phrases using TF-IDF
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Get top keywords
                top_indices = tfidf_scores.argsort()[-8:][::-1]
                top_keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0.1]
                
            except:
                # Fallback to simple frequency analysis
                words = word_tokenize(content.lower())
                word_freq = {}
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
                
                for word in words:
                    if word.isalpha() and word not in stop_words and len(word) > 3:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                top_keywords = [word for word, freq in top_keywords]
            
            # Generate creative titles based on keywords
            titles = []
            if top_keywords:
                main_keyword = top_keywords[0].title()
                
                # Creative keyword patterns
                creative_patterns = [
                    f"Mastering {main_keyword}: A Modern Approach",
                    f"Why {main_keyword} Matters More Than Ever",
                    f"The Hidden Power of {main_keyword} Revealed"
                ]
                
                # Add secondary keyword patterns if available
                if len(top_keywords) >= 2:
                    second_keyword = top_keywords[1].title()
                    creative_patterns.extend([
                        f"{main_keyword} and {second_keyword}: Perfect Together",
                        f"From {main_keyword} to {second_keyword}: A Journey"
                    ])
                
                # Format and validate titles
                for pattern in creative_patterns:
                    formatted_title = self._format_as_title(pattern)
                    if formatted_title and 3 <= len(formatted_title.split()) <= 15:
                        titles.append(formatted_title)
            
            return titles[:3]
            
        except Exception as e:
            logger.warning(f"Keyword title generation failed: {e}")
            return []
    
    def _generate_question_titles(self, content: str) -> List[str]:
        """
        Generate creative question-based titles (3-15 words)
        
        Args:
            content: Blog content
            
        Returns:
            List of engaging question titles
        """
        try:
            # Extract key phrases for context
            key_phrases = self._extract_key_phrases(content)
            if not key_phrases:
                return []
            
            main_topic = key_phrases[0]
            content_lower = content.lower()
            
            # Generate context-aware question titles
            questions = []
            
            # Tech/AI content
            if any(word in content_lower for word in ['artificial intelligence', 'ai', 'machine learning', 'technology', 'algorithm', 'data']):
                questions.extend([
                    f"Is {main_topic} the Future We've Been Waiting For?",
                    f"What Makes {main_topic} So Revolutionary?",
                    f"How Will {main_topic} Change Your Life?"
                ])
                
            # Business content
            elif any(word in content_lower for word in ['business', 'company', 'industry', 'market', 'economy', 'profit']):
                questions.extend([
                    f"Can {main_topic} Transform Your Business Strategy?",
                    f"Why Are Companies Investing in {main_topic}?",
                    f"What's the Real Impact of {main_topic}?"
                ])
                
            # Health/wellness content
            elif any(word in content_lower for word in ['health', 'wellness', 'fitness', 'medical', 'treatment']):
                questions.extend([
                    f"Could {main_topic} Be Your Health Game-Changer?",
                    f"What Do Experts Say About {main_topic}?",
                    f"Is {main_topic} Worth the Investment?"
                ])
                
            # Education/learning content
            elif any(word in content_lower for word in ['learn', 'education', 'skill', 'training', 'course', 'study']):
                questions.extend([
                    f"Why Should You Learn {main_topic} Today?",
                    f"What's the Best Way to Master {main_topic}?",
                    f"Can Anyone Really Learn {main_topic}?"
                ])
                
            # Generic but engaging questions
            else:
                questions.extend([
                    f"What's All the Hype About {main_topic}?",
                    f"Could {main_topic} Be the Answer You're Seeking?",
                    f"Why Is Everyone Talking About {main_topic}?"
                ])
            
            # Format and validate questions
            formatted_questions = []
            for question in questions:
                formatted_q = self._format_as_title(question)
                if formatted_q and 3 <= len(formatted_q.split()) <= 15:
                    if not formatted_q.endswith('?'):
                        formatted_q += '?'
                    formatted_questions.append(formatted_q)
            
            return formatted_questions[:3]
            
        except Exception as e:
            logger.warning(f"Question title generation failed: {e}")
            return []
    
    def _generate_structure_based_titles(self, content: str) -> List[str]:
        """
        Generate titles based on content structure and patterns
        
        Args:
            content: Blog content
            
        Returns:
            List of structure-based titles
        """
        try:
            sentences = sent_tokenize(content)
            if len(sentences) < 2:
                return []
            
            content_lower = content.lower()
            key_phrases = self._extract_key_phrases(content)
            
            if not key_phrases:
                return []
            
            main_topic = key_phrases[0]
            titles = []
            
            # Pattern detection and title generation
            if any(word in content_lower for word in ['step', 'guide', 'tutorial', 'process', 'method']):
                titles.extend([
                    f"A Complete Guide to {main_topic}",
                    f"Step-by-Step {main_topic} Tutorial"
                ])
            
            if any(word in content_lower for word in ['benefit', 'advantage', 'pros', 'positive']):
                titles.append(f"The Benefits of {main_topic}")
            
            if any(word in content_lower for word in ['problem', 'challenge', 'issue', 'difficulty']):
                titles.append(f"Solving {main_topic} Challenges")
            
            if any(word in content_lower for word in ['future', 'trend', 'evolution', 'development']):
                titles.append(f"The Future of {main_topic}")
            
            if any(word in content_lower for word in ['comparison', 'vs', 'versus', 'difference']):
                if len(key_phrases) >= 2:
                    titles.append(f"{key_phrases[0]} vs {key_phrases[1]}")
            
            # List-based content
            if content.count('\n-') > 2 or content.count('1.') > 0:
                titles.append(f"Essential {main_topic} Tips")
            
            return titles[:3]
            
        except Exception as e:
            logger.warning(f"Structure-based title generation failed: {e}")
            return []
    
    def _format_as_title(self, text: str) -> str:
        """
        Format text as a proper title (under 15 words)
        
        Args:
            text: Text to format
            
        Returns:
            Formatted title under 15 words
        """
        # Clean the text
        text = re.sub(r'[^\w\s.,!?:-]', '', text)
        text = text.strip()
        
        # Split into words and limit to 15 words max
        words = text.split()
        if len(words) > 15:
            words = words[:15]
        
        # Skip if too short (less than 3 words)
        if len(words) < 3:
            return ""
        
        # Articles, conjunctions, and prepositions to keep lowercase (unless first/last word)
        lowercase_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'of', 'in', 'with', 'is', 'are', 'was', 'were'}
        
        formatted_words = []
        for i, word in enumerate(words):
            if i == 0 or i == len(words) - 1 or word.lower() not in lowercase_words:
                formatted_words.append(word.capitalize())
            else:
                formatted_words.append(word.lower())
        
        title = ' '.join(formatted_words)
        
        # Ensure title is reasonable length (3-15 words, max 80 chars)
        if len(title) > 80:
            # Truncate smartly at word boundary
            words = title.split()
            truncated = []
            char_count = 0
            for word in words:
                if char_count + len(word) + 1 <= 77:  # +1 for space, 77 to leave room for "..."
                    truncated.append(word)
                    char_count += len(word) + 1
                else:
                    break
            if truncated:
                title = ' '.join(truncated) + "..."
        
        return title
    
    def _select_best_suggestions(self, suggestions: List[str], original_content: str) -> List[str]:
        """
        Select the best 3 unique title suggestions (3-15 words each)
        
        Args:
            suggestions: List of all generated suggestions
            original_content: Original blog content
            
        Returns:
            List of 3 best unique suggestions under 15 words
        """
        # Remove duplicates and validate word count
        unique_suggestions = []
        seen = set()
        
        for suggestion in suggestions:
            if suggestion and suggestion.strip():
                clean_suggestion = suggestion.strip()
                word_count = len(clean_suggestion.split())
                
                # Ensure 3-15 words and uniqueness
                if (3 <= word_count <= 15 and 
                    clean_suggestion.lower() not in seen and
                    len(clean_suggestion) > 10):  # Minimum character length
                    unique_suggestions.append(clean_suggestion)
                    seen.add(clean_suggestion.lower())
        
        # Score suggestions based on content relevance and engagement
        scored_suggestions = []
        content_words = set(word.lower() for word in word_tokenize(original_content) if word.isalpha())
        
        for suggestion in unique_suggestions:
            suggestion_words = set(word.lower() for word in word_tokenize(suggestion) if word.isalpha())
            
            # Calculate relevance score
            common_words = content_words & suggestion_words
            relevance_score = len(common_words) / max(len(suggestion_words), 1)
            
            # Bonus for engaging elements
            engagement_bonus = 0
            if any(char in suggestion for char in ['?', ':']):
                engagement_bonus += 0.15
            if any(word in suggestion.lower() for word in ['how', 'why', 'what', 'secret', 'ultimate', 'complete', 'master']):
                engagement_bonus += 0.1
            if any(word in suggestion.lower() for word in ['future', 'transform', 'revolutioniz', 'power', 'hidden']):
                engagement_bonus += 0.1
            
            # Optimal length bonus (8-12 words is ideal)
            word_count = len(suggestion.split())
            length_bonus = 0.1 if 8 <= word_count <= 12 else 0
            
            final_score = relevance_score + engagement_bonus + length_bonus
            scored_suggestions.append((suggestion, final_score))
        
        # Sort by score and take top suggestions
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top unique suggestions
        final_suggestions = [suggestion for suggestion, score in scored_suggestions[:3]]
        
        # If we still don't have 3, generate smart content-aware fallbacks
        if len(final_suggestions) < 3:
            key_phrases = self._extract_key_phrases(original_content)
            content_lower = original_content.lower()
            
            fallback_titles = []
            
            if key_phrases:
                main_topic = key_phrases[0]
                
                # Context-aware fallbacks
                if any(word in content_lower for word in ['artificial intelligence', 'ai', 'technology']):
                    fallback_titles = [
                        f"The {main_topic} Revolution: What You Need to Know",
                        f"Why {main_topic} Is Changing Everything",
                        f"Mastering {main_topic} in the Modern Age"
                    ]
                elif any(word in content_lower for word in ['business', 'company', 'market']):
                    fallback_titles = [
                        f"How {main_topic} Drives Business Success",
                        f"The Strategic Value of {main_topic}",
                        f"{main_topic}: Your Competitive Advantage"
                    ]
                else:
                    fallback_titles = [
                        f"Understanding {main_topic}: A Modern Perspective",
                        f"The Essential Guide to {main_topic}",
                        f"Why {main_topic} Matters More Than Ever"
                    ]
            else:
                fallback_titles = [
                    "Essential Insights for Modern Success",
                    "Key Strategies That Actually Work",
                    "The Complete Guide to Better Results"
                ]
            
            # Add fallbacks that meet criteria
            for fallback in fallback_titles:
                if len(final_suggestions) >= 3:
                    break
                formatted_fallback = self._format_as_title(fallback)
                word_count = len(formatted_fallback.split())
                if (3 <= word_count <= 15 and 
                    formatted_fallback.lower() not in seen):
                    final_suggestions.append(formatted_fallback)
                    seen.add(formatted_fallback.lower())
        
        return final_suggestions[:3]
    
    def _generate_additional_titles(self, content: str, existing_titles: List[str]) -> List[str]:
        """
        Generate additional content-based titles when we need more
        
        Args:
            content: Blog content
            existing_titles: Already generated titles to avoid duplicates
            
        Returns:
            List of additional unique titles
        """
        try:
            existing_lower = [title.lower() for title in existing_titles]
            key_phrases = self._extract_key_phrases(content)
            
            if not key_phrases:
                return []
            
            main_topic = key_phrases[0]
            content_lower = content.lower()
            
            # Generate more creative variations
            additional_patterns = []
            
            # Industry/domain-specific patterns
            if any(word in content_lower for word in ['artificial intelligence', 'ai', 'machine learning']):
                additional_patterns.extend([
                    f"The AI Revolution: Understanding {main_topic}",
                    f"{main_topic} and the Future of Technology",
                    f"Breaking Down {main_topic} for Everyone"
                ])
            elif any(word in content_lower for word in ['business', 'company', 'market']):
                additional_patterns.extend([
                    f"Maximizing {main_topic} for Business Growth",
                    f"The Strategic Importance of {main_topic}",
                    f"{main_topic}: A Business Game Changer"
                ])
            elif any(word in content_lower for word in ['health', 'wellness', 'fitness']):
                additional_patterns.extend([
                    f"{main_topic}: Your Path to Better Health",
                    f"The Science Behind {main_topic}",
                    f"Why {main_topic} Is Essential for Wellness"
                ])
            else:
                # Generic but engaging patterns
                additional_patterns.extend([
                    f"The Ultimate {main_topic} Resource",
                    f"Everything You Should Know About {main_topic}",
                    f"{main_topic} Made Simple and Practical"
                ])
            
            # Filter out duplicates and format
            unique_additional = []
            for pattern in additional_patterns:
                formatted_title = self._format_as_title(pattern)
                if (formatted_title and 
                    3 <= len(formatted_title.split()) <= 15 and
                    formatted_title.lower() not in existing_lower):
                    unique_additional.append(formatted_title)
                    existing_lower.append(formatted_title.lower())
                    
                    if len(unique_additional) >= 3:  # Generate up to 3 additional
                        break
            
            return unique_additional
            
        except Exception as e:
            logger.warning(f"Additional title generation failed: {e}")
            return []
    
    def get_content_hash(self, content: str) -> str:
        """
        Generate a hash of the content for caching
        
        Args:
            content: Blog content
            
        Returns:
            SHA-256 hash of the content
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
