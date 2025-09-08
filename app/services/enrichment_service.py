# app/services/enrichment_service.py
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential

# Import our database models
from app.database.model import Video, VideoEnrichment

# Import our NLP tools
from langdetect import detect, DetectorFactory
from textblob import TextBlob
import yake
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class EnrichmentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class ContentType(Enum):
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    MUSIC = "music"
    GAMING = "gaming"
    VLOG = "vlog"
    OTHER = "other"

@dataclass
class EnrichmentResult:
    """Structured result for enrichment processing"""
    video_id: int
    language: str
    sentiment: Dict[str, Any]
    keywords: List[str]
    topics: List[str]
    content_type: str
    quality_score: float
    engagement_prediction: float
    embedding: List[float]
    processing_time: float
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class ModelManager:
    """Centralized model management with lazy loading and caching"""
    
    def __init__(self):
        self._models = {}
        self._model_configs = {
            'detector_factory': lambda: self._init_detector_factory(),
            'yake_extractor': lambda: yake.KeywordExtractor(
                lan="en", n=3, deduplicationThreshold=0.7, top=15, features=None
            ),
            'embedding_model': lambda: SentenceTransformer("all-MiniLM-L6-v2"),
            'spacy_model': lambda: spacy.load("en_core_web_sm"),
            'tfidf_vectorizer': lambda: TfidfVectorizer(
                max_features=1000, stop_words='english', ngram_range=(1, 2)
            ),
        }
    
    def _init_detector_factory(self):
        DetectorFactory.seed = 0
        return DetectorFactory
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_model(self, model_name: str):
        """Get model with lazy loading and retry logic"""
        if model_name not in self._models:
            if model_name not in self._model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            try:
                logger.info(f"Loading model: {model_name}")
                self._models[model_name] = self._model_configs[model_name]()
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        
        return self._models[model_name]
    
    def preload_all_models(self):
        """Preload all models for better performance"""
        for model_name in self._model_configs.keys():
            try:
                self.get_model(model_name)
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")

# Global model manager instance
model_manager = ModelManager()

class CacheManager:
    """Redis-based cache manager for enrichment results"""
    
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis = redis_client
        self.cache_ttl = 86400 * 7  # 7 days
    
    def _get_cache_key(self, video_id: int, content_hash: str) -> str:
        return f"enrichment:{video_id}:{content_hash}"
    
    def _get_content_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_cached_result(self, video_id: int, content: str) -> Optional[EnrichmentResult]:
        if not self.redis:
            return None
        
        try:
            content_hash = self._get_content_hash(content)
            cache_key = self._get_cache_key(video_id, content_hash)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return EnrichmentResult(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def cache_result(self, video_id: int, content: str, result: EnrichmentResult):
        if not self.redis:
            return
        
        try:
            content_hash = self._get_content_hash(content)
            cache_key = self._get_cache_key(video_id, content_hash)
            
            # Convert dataclass to dict for JSON serialization
            result_dict = {
                'video_id': result.video_id,
                'language': result.language,
                'sentiment': result.sentiment,
                'keywords': result.keywords,
                'topics': result.topics,
                'content_type': result.content_type,
                'quality_score': result.quality_score,
                'engagement_prediction': result.engagement_prediction,
                'embedding': result.embedding,
                'processing_time': result.processing_time,
                'confidence_scores': result.confidence_scores,
                'metadata': result.metadata
            }
            
            await self.redis.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(result_dict, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

class AdvancedNLPProcessor:
    """Advanced NLP processing with multiple techniques"""
    
    def __init__(self):
        self.content_type_keywords = {
            ContentType.EDUCATIONAL: ["learn", "tutorial", "education", "course", "lesson", "guide"],
            ContentType.ENTERTAINMENT: ["funny", "comedy", "entertainment", "fun", "laugh"],
            ContentType.NEWS: ["news", "breaking", "update", "report", "current"],
            ContentType.TUTORIAL: ["how to", "tutorial", "step by step", "guide", "walkthrough"],
            ContentType.REVIEW: ["review", "unboxing", "test", "comparison", "rating"],
            ContentType.MUSIC: ["music", "song", "album", "artist", "concert"],
            ContentType.GAMING: ["game", "gaming", "play", "gameplay", "stream"],
            ContentType.VLOG: ["vlog", "daily", "life", "personal", "diary"]
        }
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Enhanced language detection with confidence score"""
        try:
            if not text or len(text.strip()) < 10:
                return "unknown", 0.0
            
            # Clean text for better detection
            clean_text = self._clean_text_for_language_detection(text)
            language = detect(clean_text)
            
            # Simple confidence estimation based on text length
            confidence = min(len(clean_text) / 100, 1.0)
            return language, confidence
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown", 0.0
    
    def _clean_text_for_language_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        # Remove URLs, mentions, hashtags
        import re
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        # Keep only alphanumeric and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        return text.strip()
    
    def analyze_sentiment_advanced(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis with multiple metrics"""
        try:
            # TextBlob baseline
            blob = TextBlob(text[:2000])  # Limit text length
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Enhanced sentiment categorization
            if polarity > 0.3:
                label = "very_positive"
            elif polarity > 0.1:
                label = "positive"
            elif polarity > -0.1:
                label = "neutral"
            elif polarity > -0.3:
                label = "negative"
            else:
                label = "very_negative"
            
            # Calculate confidence based on subjectivity
            confidence = abs(polarity) * (1 - subjectivity)
            
            return {
                "label": label,
                "polarity": float(polarity),
                "subjectivity": float(subjectivity),
                "confidence": float(confidence),
                "intensity": abs(polarity)
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {
                "label": "neutral",
                "polarity": 0.0,
                "subjectivity": 0.0,
                "confidence": 0.0,
                "intensity": 0.0
            }
    
    def extract_keywords_advanced(self, text: str) -> Tuple[List[str], Dict[str, float]]:
        """Advanced keyword extraction using multiple methods"""
        keywords = []
        scores = {}
        
        try:
            # YAKE extraction
            yake_extractor = model_manager.get_model('yake_extractor')
            yake_keywords = yake_extractor.extract_keywords(text)
            
            # spaCy NER and important terms
            try:
                nlp = model_manager.get_model('spacy_model')
                doc = nlp(text[:5000])  # Limit for performance
                
                # Extract named entities
                entities = [ent.text.lower() for ent in doc.ents 
                           if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
                
                # Extract important nouns and adjectives
                important_terms = [token.lemma_.lower() for token in doc 
                                 if token.pos_ in ['NOUN', 'ADJ'] and 
                                 len(token.lemma_) > 3 and
                                 not token.is_stop]
                
                # Combine all sources
                all_terms = []
                for kw, score in yake_keywords:
                    all_terms.append(kw.lower())
                    scores[kw.lower()] = 1.0 - score  # YAKE gives lower scores for better terms
                
                for term in entities + important_terms:
                    if term not in scores:
                        scores[term] = 0.5  # Default score for NLP-extracted terms
                    all_terms.append(term)
                
                # Remove duplicates while preserving order
                keywords = list(dict.fromkeys(all_terms))[:15]
                
            except Exception as e:
                logger.warning(f"spaCy processing failed, falling back to YAKE only: {e}")
                keywords = [kw for kw, score in yake_keywords]
                scores = {kw: 1.0 - score for kw, score in yake_keywords}
        
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
        
        return keywords, scores
    
    def classify_content_type(self, title: str, description: str, keywords: List[str]) -> Tuple[ContentType, float]:
        """Classify content type based on text analysis"""
        try:
            text = f"{title} {description}".lower()
            
            # Score each content type
            type_scores = {}
            for content_type, type_keywords in self.content_type_keywords.items():
                score = 0
                for keyword in type_keywords:
                    score += text.count(keyword)
                    # Bonus for keywords in extracted keywords
                    if any(keyword in kw for kw in keywords):
                        score += 2
                type_scores[content_type] = score
            
            # Find best match
            if not type_scores or max(type_scores.values()) == 0:
                return ContentType.OTHER, 0.0
            
            best_type = max(type_scores, key=type_scores.get)
            confidence = min(type_scores[best_type] / 5.0, 1.0)  # Normalize confidence
            
            return best_type, confidence
            
        except Exception as e:
            logger.warning(f"Content classification failed: {e}")
            return ContentType.OTHER, 0.0
    
    def calculate_quality_score(self, title: str, description: str, 
                              keywords: List[str], sentiment: Dict[str, Any]) -> float:
        """Calculate content quality score based on multiple factors"""
        try:
            score = 0.0
            
            # Title quality (0-25 points)
            if title:
                title_len = len(title)
                if 10 <= title_len <= 100:  # Optimal length
                    score += 25 * (1 - abs(title_len - 55) / 45)  # Peak at 55 chars
                else:
                    score += 10
            
            # Description quality (0-25 points)
            if description:
                desc_len = len(description)
                if desc_len > 50:
                    score += 25 * min(desc_len / 500, 1.0)  # Up to 500 chars for full score
            
            # Keyword richness (0-25 points)
            if keywords:
                score += min(len(keywords) * 2, 25)
            
            # Sentiment positivity (0-25 points)
            sentiment_bonus = max(sentiment.get('polarity', 0) * 25, 0)
            score += sentiment_bonus
            
            return min(score / 100, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5
    
    def predict_engagement(self, title: str, description: str, 
                          sentiment: Dict[str, Any], quality_score: float) -> float:
        """Predict engagement potential based on content analysis"""
        try:
            engagement = 0.0
            
            # Title engagement factors
            if title:
                # Question marks and exclamation points
                engagement += title.count('?') * 0.1
                engagement += title.count('!') * 0.1
                
                # Engaging words
                engaging_words = ['amazing', 'incredible', 'shocking', 'secret', 'revealed']
                for word in engaging_words:
                    if word in title.lower():
                        engagement += 0.1
            
            # Sentiment impact
            sentiment_impact = abs(sentiment.get('polarity', 0)) * 0.3
            engagement += sentiment_impact
            
            # Quality score impact
            engagement += quality_score * 0.4
            
            # Normalize to 0-1
            return min(engagement, 1.0)
            
        except Exception as e:
            logger.warning(f"Engagement prediction failed: {e}")
            return 0.5

class EnrichmentService:
    """
    Enhanced service for video content enrichment with advanced NLP capabilities
    """
    
    def __init__(self, db_session: AsyncSession, redis_client: Optional[Any] = None):
        self.db = db_session
        self.cache_manager = CacheManager(redis_client)
        self.nlp_processor = AdvancedNLPProcessor()
    
    async def enrich_video(self, video_id: int, force_refresh: bool = False) -> EnrichmentResult:
        """
        Main enrichment pipeline with caching and comprehensive analysis
        """
        start_time = datetime.now()
        logger.info(f"Starting enrichment for video_id: {video_id}")
        
        # 1. Fetch video data
        video = await self._get_video_with_validation(video_id)
        content = self._prepare_content(video)
        
        # 2. Check cache if not forcing refresh
        if not force_refresh:
            cached_result = await self.cache_manager.get_cached_result(video_id, content)
            if cached_result:
                logger.info(f"Using cached result for video_id: {video_id}")
                cached_result.metadata['status'] = EnrichmentStatus.CACHED.value
                return cached_result
        
        try:
            # 3. Run comprehensive NLP analysis
            result = await self._run_enrichment_pipeline(video_id, video, content)
            
            # 4. Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.metadata['status'] = EnrichmentStatus.COMPLETED.value
            result.metadata['processed_at'] = datetime.now().isoformat()
            
            # 5. Save to database
            await self._save_enrichment_result(result)
            
            # 6. Cache the result
            await self.cache_manager.cache_result(video_id, content, result)
            
            logger.info(f"Successfully completed enrichment for video_id: {video_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Enrichment failed for video_id: {video_id}: {e}", exc_info=True)
            # Return a basic result with error information
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_error_result(video_id, str(e), processing_time)
    
    async def _get_video_with_validation(self, video_id: int) -> Video:
        """Fetch and validate video exists"""
        result = await self.db.execute(select(Video).where(Video.id == video_id))
        video = result.scalar_one_or_none()
        
        if not video:
            raise ValueError(f"Video with id {video_id} not found")
        
        return video
    
    def _prepare_content(self, video: Video) -> str:
        """Prepare content for analysis"""
        title = video.title or ""
        description = video.description or ""
        
        # Add any additional metadata that might be useful
        additional_info = []
        if hasattr(video, 'tags') and video.tags:
            additional_info.append(f"Tags: {', '.join(video.tags)}")
        
        content_parts = [title, description] + additional_info
        return "\n\n".join(part for part in content_parts if part.strip())
    
    async def _run_enrichment_pipeline(self, video_id: int, video: Video, content: str) -> EnrichmentResult:
        """Run the complete enrichment pipeline"""
        
        # Language detection
        language, lang_confidence = self.nlp_processor.detect_language(content)
        
        # Sentiment analysis
        sentiment = self.nlp_processor.analyze_sentiment_advanced(content)
        
        # Keyword extraction
        keywords, keyword_scores = self.nlp_processor.extract_keywords_advanced(content)
        
        # Content type classification
        content_type, type_confidence = self.nlp_processor.classify_content_type(
            video.title or "", video.description or "", keywords
        )
        
        # Quality score calculation
        quality_score = self.nlp_processor.calculate_quality_score(
            video.title or "", video.description or "", keywords, sentiment
        )
        
        # Engagement prediction
        engagement_prediction = self.nlp_processor.predict_engagement(
            video.title or "", video.description or "", sentiment, quality_score
        )
        
        # Generate embedding
        embedding = await self._generate_embedding_async(content)
        
        # Topic extraction (using clustering on keywords)
        topics = self._extract_topics(keywords, keyword_scores)
        
        # Confidence scores
        confidence_scores = {
            'language': lang_confidence,
            'sentiment': sentiment.get('confidence', 0.0),
            'content_type': type_confidence,
            'overall': (lang_confidence + sentiment.get('confidence', 0.0) + type_confidence) / 3
        }
        
        # Metadata
        metadata = {
            'model_versions': {
                'langdetect': '1.0.9',
                'textblob': '0.17.1',
                'sentence_transformers': '2.2.2',
                'yake': '0.4.8'
            },
            'content_length': len(content),
            'keyword_count': len(keywords),
            'has_description': bool(video.description)
        }
        
        return EnrichmentResult(
            video_id=video_id,
            language=language,
            sentiment=sentiment,
            keywords=keywords[:10],  # Limit to top 10
            topics=topics,
            content_type=content_type.value,
            quality_score=quality_score,
            engagement_prediction=engagement_prediction,
            embedding=embedding,
            processing_time=0.0,  # Will be set later
            confidence_scores=confidence_scores,
            metadata=metadata
        )
    
    async def _generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding asynchronously"""
        try:
            embedding_model = model_manager.get_model('embedding_model')
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: embedding_model.encode([text[:4096]])[0]
            )
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return []
    
    def _extract_topics(self, keywords: List[str], keyword_scores: Dict[str, float]) -> List[str]:
        """Extract high-level topics from keywords"""
        if not keywords:
            return []
        
        try:
            # Simple topic grouping based on semantic similarity
            # In a real implementation, you might use more sophisticated clustering
            topics = []
            
            # Group similar keywords
            keyword_groups = {
                'technology': ['tech', 'software', 'app', 'digital', 'computer', 'ai', 'data'],
                'business': ['business', 'marketing', 'sales', 'finance', 'entrepreneur'],
                'entertainment': ['movie', 'music', 'game', 'fun', 'comedy', 'show'],
                'education': ['learn', 'education', 'tutorial', 'guide', 'course'],
                'health': ['health', 'fitness', 'medical', 'wellness', 'exercise'],
                'lifestyle': ['lifestyle', 'travel', 'food', 'fashion', 'home']
            }
            
            for topic, topic_keywords in keyword_groups.items():
                if any(kw in ' '.join(keywords).lower() for kw in topic_keywords):
                    topics.append(topic)
            
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []
    
    async def _save_enrichment_result(self, result: EnrichmentResult):
        """Save enrichment result to database"""
        try:
            enrichment_data = VideoEnrichment(
                video_id=result.video_id,
                sentiment=result.sentiment['label'],
                topics=result.topics,
                embedding=result.embedding,
                # Add new fields based on enhanced analysis
                language=result.language,
                keywords=result.keywords,
                content_type=result.content_type,
                quality_score=result.quality_score,
                engagement_prediction=result.engagement_prediction,
                confidence_scores=result.confidence_scores,
                metadata=result.metadata
            )
            
            self.db.add(enrichment_data)
            await self.db.commit()
            await self.db.refresh(enrichment_data)
            
        except Exception as e:
            logger.error(f"Failed to save enrichment result: {e}")
            await self.db.rollback()
            raise
    
    def _create_error_result(self, video_id: int, error_message: str, processing_time: float) -> EnrichmentResult:
        """Create a basic result for failed enrichment"""
        return EnrichmentResult(
            video_id=video_id,
            language="unknown",
            sentiment={"label": "neutral", "polarity": 0.0, "confidence": 0.0},
            keywords=[],
            topics=[],
            content_type=ContentType.OTHER.value,
            quality_score=0.0,
            engagement_prediction=0.0,
            embedding=[],
            processing_time=processing_time,
            confidence_scores={"overall": 0.0},
            metadata={
                "status": EnrichmentStatus.FAILED.value,
                "error": error_message,
                "processed_at": datetime.now().isoformat()
            }
        )
    
    async def batch_enrich_videos(self, video_ids: List[int], max_concurrent: int = 5) -> List[EnrichmentResult]:
        """Enrich multiple videos concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def enrich_with_semaphore(video_id: int):
            async with semaphore:
                return await self.enrich_video(video_id)
        
        tasks = [enrich_with_semaphore(video_id) for video_id in video_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    self._create_error_result(video_ids[i], str(result), 0.0)
                )
            else:
                processed_results.append(result)
        
        return processed_results

# Initialize models on import (optional - can be done on first use instead)
def initialize_models():
    """Initialize all models for better startup performance"""
    try:
        model_manager.preload_all_models()
        logger.info("All NLP models preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")

# Uncomment to preload models on import
# initialize_models()