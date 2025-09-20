# app/services/enrichment_service.py
import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Import our database models
from app.database.models import Video, VideoEnrichment
from app.config.base import settings
from app.exceptions import EnrichmentError, ModelLoadError, ValidationError

# Import our NLP tools
from langdetect import detect, DetectorFactory
from textblob import TextBlob
import yake
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Monitoring and metrics
ENRICHMENT_REQUESTS = Counter('enrichment_requests_total', 'Total enrichment requests', ['status'])
ENRICHMENT_DURATION = Histogram('enrichment_duration_seconds', 'Time spent on enrichment')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')
ACTIVE_ENRICHMENTS = Gauge('active_enrichments', 'Number of active enrichment processes')
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Time spent loading models', ['model_name'])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnrichmentStatus(Enum):
    """Status enum for enrichment processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"

class ContentType(Enum):
    """Enhanced content type classification with confidence scoring"""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    MUSIC = "music"
    GAMING = "gaming"
    VLOG = "vlog"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    HEALTH_FITNESS = "health_fitness"
    COOKING = "cooking"
    TRAVEL = "travel"
    SCIENCE = "science"
    SPORTS = "sports"
    DIY_CRAFTS = "diy_crafts"
    FASHION_BEAUTY = "fashion_beauty"
    PETS_ANIMALS = "pets_animals"
    COMEDY = "comedy"
    DOCUMENTARY = "documentary"
    OTHER = "other"

class ProcessingPriority(Enum):
    """Processing priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class EnrichmentConfig:
    """Configuration for enrichment processing"""
    max_text_length: int = 10000
    max_keywords: int = 15
    max_topics: int = 8
    embedding_max_length: int = 4096
    sentiment_confidence_threshold: float = 0.3
    language_confidence_threshold: float = 0.5
    cache_ttl_seconds: int = 86400 * 7  # 7 days
    timeout_seconds: int = 300  # 5 minutes
    retry_attempts: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 20

@dataclass
class EnrichmentResult:
    """Comprehensive enrichment result with metadata"""
    video_id: int
    request_id: str
    language: str
    language_confidence: float
    sentiment: Dict[str, Any]
    keywords: List[Dict[str, Any]]  # Enhanced with scores and categories
    topics: List[Dict[str, Any]]  # Enhanced with confidence scores
    content_type: str
    content_type_confidence: float
    quality_score: float
    engagement_prediction: float
    readability_score: float
    embedding: List[float]
    processing_time: float
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class ContentTypeClassifier:
    """Advanced content type classifier with machine learning capabilities"""
    
    def __init__(self):
        self.content_type_patterns = {
            ContentType.EDUCATIONAL: {
                'keywords': ['learn', 'education', 'course', 'lesson', 'study', 'academic', 'knowledge', 'teach'],
                'title_patterns': [r'how to', r'learn.*', r'.*course', r'.*lesson', r'explained'],
                'weight': 1.2
            },
            ContentType.TUTORIAL: {
                'keywords': ['tutorial', 'how to', 'step by step', 'guide', 'walkthrough', 'instructions', 'demo'],
                'title_patterns': [r'how to.*', r'.*tutorial', r'step.*step', r'.*guide'],
                'weight': 1.3
            },
            ContentType.REVIEW: {
                'keywords': ['review', 'unboxing', 'test', 'comparison', 'rating', 'opinion', 'verdict', 'pros', 'cons'],
                'title_patterns': [r'.*review', r'unboxing.*', r'.*vs.*', r'.*comparison'],
                'weight': 1.1
            },
            ContentType.ENTERTAINMENT: {
                'keywords': ['funny', 'comedy', 'entertainment', 'fun', 'laugh', 'hilarious', 'amusing'],
                'title_patterns': [r'funny.*', r'.*comedy', r'hilarious.*', r'.*fails'],
                'weight': 1.0
            },
            ContentType.NEWS: {
                'keywords': ['news', 'breaking', 'update', 'report', 'current', 'today', 'latest', 'headline'],
                'title_patterns': [r'breaking.*', r'.*news', r'latest.*', r'update.*'],
                'weight': 1.4
            },
            ContentType.MUSIC: {
                'keywords': ['music', 'song', 'album', 'artist', 'concert', 'lyrics', 'band', 'melody'],
                'title_patterns': [r'.*music', r'.*song', r'.*album', r'.*concert'],
                'weight': 1.2
            },
            ContentType.GAMING: {
                'keywords': ['game', 'gaming', 'play', 'gameplay', 'stream', 'gamer', 'walkthrough', 'let\'s play'],
                'title_patterns': [r'.*gameplay', r'gaming.*', r'let.*play', r'.*walkthrough'],
                'weight': 1.2
            },
            ContentType.VLOG: {
                'keywords': ['vlog', 'daily', 'life', 'personal', 'diary', 'day in', 'routine', 'lifestyle'],
                'title_patterns': [r'vlog.*', r'day.*life', r'daily.*', r'.*routine'],
                'weight': 1.0
            },
            ContentType.TECHNOLOGY: {
                'keywords': ['tech', 'technology', 'gadget', 'device', 'software', 'app', 'digital', 'AI', 'computer'],
                'title_patterns': [r'tech.*', r'.*technology', r'.*review.*tech', r'AI.*'],
                'weight': 1.1
            },
            ContentType.COOKING: {
                'keywords': ['cooking', 'recipe', 'food', 'kitchen', 'chef', 'ingredients', 'baking', 'meal'],
                'title_patterns': [r'recipe.*', r'cooking.*', r'.*food', r'how.*cook'],
                'weight': 1.2
            },
            ContentType.HEALTH_FITNESS: {
                'keywords': ['fitness', 'workout', 'health', 'exercise', 'gym', 'training', 'nutrition', 'wellness'],
                'title_patterns': [r'workout.*', r'fitness.*', r'.*exercise', r'health.*'],
                'weight': 1.1
            },
            ContentType.TRAVEL: {
                'keywords': ['travel', 'trip', 'vacation', 'destination', 'journey', 'adventure', 'explore'],
                'title_patterns': [r'travel.*', r'.*trip', r'vacation.*', r'.*destination'],
                'weight': 1.1
            }
        }
        
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        import re
        for content_type, data in self.content_type_patterns.items():
            self._compiled_patterns[content_type] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in data.get('title_patterns', [])
            ]
    
    def classify(self, title: str, description: str, keywords: List[str]) -> Tuple[ContentType, float]:
        """Classify content type with confidence score"""
        try:
            text = f"{title} {description}".lower()
            keyword_text = ' '.join(keywords).lower()
            
            type_scores = {}
            
            for content_type, patterns in self.content_type_patterns.items():
                score = 0.0
                
                # Keyword matching
                for keyword in patterns['keywords']:
                    # Higher weight for title matches
                    if keyword in title.lower():
                        score += 3.0
                    if keyword in description.lower():
                        score += 2.0
                    if keyword in keyword_text:
                        score += 1.5
                
                # Pattern matching in title
                for pattern in self._compiled_patterns.get(content_type, []):
                    if pattern.search(title):
                        score += 4.0
                    if pattern.search(description):
                        score += 2.0
                
                # Apply content type weight
                score *= patterns.get('weight', 1.0)
                type_scores[content_type] = score
            
            if not type_scores or max(type_scores.values()) == 0:
                return ContentType.OTHER, 0.0
            
            best_type = max(type_scores, key=type_scores.get)
            max_score = type_scores[best_type]
            
            # Calculate confidence (0-1 scale)
            confidence = min(max_score / 10.0, 1.0)
            
            # Apply confidence threshold
            if confidence < 0.3:
                return ContentType.OTHER, confidence
            
            return best_type, confidence
            
        except Exception as e:
            logger.warning(f"Content classification failed: {e}")
            return ContentType.OTHER, 0.0

class ModelManager:
    """Production-ready model manager with health monitoring and lazy loading"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._model_configs = {
            'detector_factory': {
                # FIX 1: Accept the unused argument from the executor
                'loader': lambda *args, **kwargs: self._init_detector_factory(),
                'health_check': lambda model: True,
                'timeout': 30
            },
            'yake_extractor': {
                # FIX 2: Use the correct parameter name 'dedupLim'
                'loader': lambda: yake.KeywordExtractor(
                    lan="en", n=3, dedupLim=0.7, top=15, features=None
                ),
                'health_check': lambda model: hasattr(model, 'extract_keywords'),
                'timeout': 30
            },
            'embedding_model': {
                'loader': lambda: SentenceTransformer("all-MiniLM-L6-v2"),
                'health_check': lambda model: hasattr(model, 'encode'),
                'timeout': 120
            },
            'spacy_model': {
                'loader': lambda: spacy.load("en_core_web_sm"),
                'health_check': lambda model: model.has_pipe('ner'),
                'timeout': 60
            },
            'tfidf_vectorizer': {
                'loader': lambda: TfidfVectorizer(
                    max_features=1000, stop_words='english', ngram_range=(1, 2)
                ),
                'health_check': lambda model: hasattr(model, 'fit_transform'),
                'timeout': 30
            }
        }
        self._lock = threading.RLock()
        self._model_health: Dict[str, bool] = {}
        self._last_health_check: Dict[str, datetime] = {}
        
    def _init_detector_factory(self):
        """Initialize language detector factory"""
        DetectorFactory.seed = 0
        return DetectorFactory
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ImportError, OSError, ModelLoadError))
    )
    def get_model(self, model_name: str, force_reload: bool = False):
        """Get model with comprehensive error handling and health checks."""
        with self._lock:
            if model_name not in self._model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            if force_reload or model_name not in self._models:
                with MODEL_LOAD_TIME.labels(model_name=model_name).time():
                    try:
                        logger.info(f"Loading model: {model_name}")
                        config = self._model_configs[model_name]
                        
                        # This new block is cross-platform and more robust.
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(config['loader'])
                            try:
                                model = future.result(timeout=config.get('timeout', 60))
                            except FuturesTimeoutError:
                                raise ModelLoadError(f"Model {model_name} loading timed out")

                        if not config['health_check'](model):
                            raise ModelLoadError(f"Model {model_name} failed health check")
                        
                        self._models[model_name] = model
                        self._model_health[model_name] = True
                        self._last_health_check[model_name] = datetime.now()
                        
                        logger.info(f"Successfully loaded model: {model_name}")
                        
                    except Exception as e:
                        self._model_health[model_name] = False
                        logger.error(f"Failed to load model {model_name}: {e}")
                        raise ModelLoadError(f"Failed to load {model_name}: {str(e)}")
            
            # Periodic health check (every hour)
            if (model_name in self._last_health_check and 
                datetime.now() - self._last_health_check[model_name] > timedelta(hours=1)):
                try:
                    config = self._model_configs[model_name]
                    if not config['health_check'](self._models[model_name]):
                        logger.warning(f"Model {model_name} failed health check, reloading...")
                        return self.get_model(model_name, force_reload=True)
                    self._last_health_check[model_name] = datetime.now()
                except Exception as e:
                    logger.warning(f"Health check failed for {model_name}: {e}")
            
            return self._models[model_name]
    
    def get_model_health(self) -> Dict[str, bool]:
        """Get health status of all models"""
        return self._model_health.copy()
    
    async def preload_all_models(self):
        """Preload all models asynchronously."""
        loop = asyncio.get_event_loop()
        tasks = []
        for model_name in self._model_configs.keys():
            # Schedule the synchronous get_model function to run in the default executor
            task = loop.run_in_executor(
                None, self.get_model, model_name
            )
            tasks.append(task)
        
        # asyncio.gather can correctly await a list of Futures.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            model_name = list(self._model_configs.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to preload model {model_name}: {result}")

# Global model manager instance
model_manager = ModelManager()

class CacheManager:
    """Production Redis cache manager with connection pooling and failover"""
    
    def __init__(self, redis_url: Optional[str] = None, config: EnrichmentConfig = None):
        self.config = config or EnrichmentConfig()
        self.redis_pool = None
        self._initialize_redis(redis_url)
    
    def _initialize_redis(self, redis_url: Optional[str]):
        """Initialize Redis connection with error handling"""
        try:
            if redis_url or getattr(settings, 'REDIS_URL', None):
                self.redis_pool = redis.ConnectionPool.from_url(
                    redis_url or settings.REDIS_URL,
                    max_connections=20,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                logger.info("Redis cache initialized")
            else:
                logger.warning("Redis URL not provided, caching disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_pool = None
    
    def _get_cache_key(self, video_id: int, content_hash: str) -> str:
        """Generate cache key"""
        return f"enrichment:v2:{video_id}:{content_hash}"
    
    def _get_content_hash(self, content: str) -> str:
        """Generate content hash for cache key"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get_cached_result(self, video_id: int, content: str) -> Optional[EnrichmentResult]:
        """Get cached enrichment result with error handling"""
        if not self.redis_pool:
            return None
        
        try:
            async with redis.Redis(connection_pool=self.redis_pool) as client:
                content_hash = self._get_content_hash(content)
                cache_key = self._get_cache_key(video_id, content_hash)
                
                cached_data = await client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    # Convert datetime string back to datetime object
                    if 'created_at' in data:
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                    
                    CACHE_HITS.inc()
                    return EnrichmentResult(**data)
                
                CACHE_MISSES.inc()
                return None
                
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def cache_result(self, video_id: int, content: str, result: EnrichmentResult):
        """Cache enrichment result with error handling"""
        if not self.redis_pool:
            return
        
        try:
            async with redis.Redis(connection_pool=self.redis_pool) as client:
                content_hash = self._get_content_hash(content)
                cache_key = self._get_cache_key(video_id, content_hash)
                
                # Convert result to dict and handle datetime serialization
                result_dict = result.to_dict()
                result_dict['created_at'] = result.created_at.isoformat()
                
                await client.setex(
                    cache_key,
                    self.config.cache_ttl_seconds,
                    json.dumps(result_dict, default=str)
                )
                
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def invalidate_cache(self, video_id: int):
        """Invalidate all cache entries for a video"""
        if not self.redis_pool:
            return
        
        try:
            async with redis.Redis(connection_pool=self.redis_pool) as client:
                pattern = f"enrichment:v2:{video_id}:*"
                keys = await client.keys(pattern)
                if keys:
                    await client.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cache entries for video {video_id}")
                    
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_pool:
            return {"status": "disabled"}
        
        try:
            async with redis.Redis(connection_pool=self.redis_pool) as client:
                info = await client.info('memory')
                return {
                    "status": "enabled",
                    "used_memory": info.get('used_memory_human', 'unknown'),
                    "connected_clients": info.get('connected_clients', 0),
                }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

class AdvancedNLPProcessor:
    """Production NLP processor with enhanced analysis capabilities"""
    
    def __init__(self, config: EnrichmentConfig = None):
        self.config = config or EnrichmentConfig()
        self.content_classifier = ContentTypeClassifier()
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nlp-worker")
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Enhanced language detection with confidence scoring"""
        try:
            if not text or len(text.strip()) < 10:
                return "unknown", 0.0
            
            clean_text = self._clean_text_for_language_detection(text)
            if len(clean_text) < 10:
                return "unknown", 0.0
            
            language = detect(clean_text)
            
            # Enhanced confidence calculation
            text_length = len(clean_text)
            base_confidence = min(text_length / 200, 1.0)
            
            # Bonus for longer text and common languages
            if language in ['en', 'es', 'fr', 'de', 'it']:
                base_confidence *= 1.2
            
            confidence = min(base_confidence, 1.0)
            return language, confidence
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown", 0.0
    
    def _clean_text_for_language_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        import re
        
        # Remove URLs, mentions, hashtags, and special characters
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s.,!?-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def analyze_sentiment_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis with multiple metrics"""
        try:
            # Limit text length for processing
            text = text[:self.config.max_text_length]
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Enhanced sentiment categorization with more granular labels
            if polarity > 0.5:
                label = "very_positive"
            elif polarity > 0.2:
                label = "positive"
            elif polarity > 0.05:
                label = "slightly_positive"
            elif polarity > -0.05:
                label = "neutral"
            elif polarity > -0.2:
                label = "slightly_negative"
            elif polarity > -0.5:
                label = "negative"
            else:
                label = "very_negative"
            
            # Calculate confidence based on polarity strength and text objectivity
            confidence = min(abs(polarity) * (1 - subjectivity * 0.3), 1.0)
            
            # Additional sentiment features
            intensity = abs(polarity)
            emotional_range = "high" if intensity > 0.6 else "medium" if intensity > 0.3 else "low"
            
            return {
                "label": label,
                "polarity": float(polarity),
                "subjectivity": float(subjectivity),
                "confidence": float(confidence),
                "intensity": float(intensity),
                "emotional_range": emotional_range,
                "is_objective": subjectivity < 0.3,
                "is_emotional": intensity > 0.4
            }
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {
                "label": "neutral",
                "polarity": 0.0,
                "subjectivity": 0.0,
                "confidence": 0.0,
                "intensity": 0.0,
                "emotional_range": "low",
                "is_objective": True,
                "is_emotional": False
            }
    
    def extract_keywords_advanced(self, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Advanced keyword extraction with categorization and scoring"""
        keywords = []
        all_scores = {}
        
        try:
            # Limit text length
            text = text[:self.config.max_text_length]
            
            # YAKE extraction
            yake_extractor = model_manager.get_model('yake_extractor')
            yake_keywords = yake_extractor.extract_keywords(text)
            
            # spaCy NER and POS analysis
            try:
                nlp = model_manager.get_model('spacy_model')
                doc = nlp(text)
                
                # Extract named entities with categories
                entities = []
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        entities.append({
                            'text': ent.text.lower().strip(),
                            'category': 'entity',
                            'entity_type': ent.label_,
                            'score': 0.8
                        })
                
                # Extract important terms with POS tagging
                important_terms = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and 
                        len(token.lemma_) > 2 and
                        not token.is_stop and
                        not token.is_punct and
                        token.is_alpha):
                        
                        important_terms.append({
                            'text': token.lemma_.lower(),
                            'category': 'term',
                            'pos': token.pos_,
                            'score': 0.6 if token.pos_ == 'NOUN' else 0.4
                        })
                
                # Process YAKE keywords
                yake_terms = []
                for kw, score in yake_keywords:
                    yake_terms.append({
                        'text': kw.lower().strip(),
                        'category': 'keyword',
                        'score': max(1.0 - score, 0.1),  # YAKE gives lower scores for better terms
                        'extraction_method': 'yake'
                    })
                
                # Combine and deduplicate
                all_keywords = yake_terms + entities + important_terms
                keyword_dict = {}
                
                for kw in all_keywords:
                    text = kw['text']
                    if text in keyword_dict:
                        # Merge with higher score
                        if kw['score'] > keyword_dict[text]['score']:
                            keyword_dict[text].update(kw)
                    else:
                        keyword_dict[text] = kw
                
                # Sort by score and limit
                sorted_keywords = sorted(
                    keyword_dict.values(), 
                    key=lambda x: x['score'], 
                    reverse=True
                )
                
                keywords = sorted_keywords[:self.config.max_keywords]
                all_scores = {kw['text']: kw['score'] for kw in keywords}
                
            except Exception as e:
                logger.warning(f"spaCy processing failed, using YAKE only: {e}")
                # Fallback to YAKE only
                for kw, score in yake_keywords[:self.config.max_keywords]:
                    keywords.append({
                        'text': kw.lower().strip(),
                        'category': 'keyword',
                        'score': max(1.0 - score, 0.1),
                        'extraction_method': 'yake'
                    })
                    all_scores[kw.lower().strip()] = max(1.0 - score, 0.1)
        
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
        
        return keywords, all_scores
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate readability score using multiple metrics"""
        try:
            if not text or len(text.strip()) < 10:
                return 0.5
            
            # Simple readability metrics
            sentences = text.split('.')
            words = text.split()
            
            if not sentences or not words:
                return 0.5
            
            avg_words_per_sentence = len(words) / len(sentences)
            avg_chars_per_word = sum(len(word) for word in words) / len(words)
            
            # Flesch-like score (simplified)
            readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * (avg_chars_per_word / 4.7))
            
            # Normalize to 0-1 scale
            normalized_score = max(0, min(1, (readability + 100) / 200))
            
            return float(normalized_score)
            
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 0.5
    
    def calculate_quality_score(self, title: str, description: str, 
                              keywords: List[Dict[str, Any]], sentiment: Dict[str, Any],
                              readability: float) -> float:
        """Enhanced quality score calculation with multiple factors"""
        try:
            score = 0.0
            
            # Title quality (0-20 points)
            if title:
                title_len = len(title)
                if 20 <= title_len <= 80:  # Optimal length
                    score += 20 * (1 - abs(title_len - 50) / 30)
                elif 10 <= title_len <= 120:
                    score += 15
                else:
                    score += 5
                
                # Title engagement factors
                if any(char in title for char in '?!'):
                    score += 2
                if any(word in title.lower() for word in ['how', 'why', 'what', 'best', 'top']):
                    score += 3
            
            # Description quality (0-25 points)
            if description:
                desc_len = len(description)
                if desc_len > 100:
                    score += 25 * min(desc_len / 1000, 1.0)
                elif desc_len > 50:
                    score += 15
                else:
                    score += 5
            
            # Keyword richness (0-20 points)
            if keywords:
                keyword_count = len(keywords)
                avg_keyword_score = sum(kw.get('score', 0) for kw in keywords) / len(keywords)
                score += min(keyword_count * 2, 15) + (avg_keyword_score * 5)
            
            # Sentiment quality (0-15 points)
            sentiment_confidence = sentiment.get('confidence', 0)
            if sentiment.get('label') in ['positive', 'very_positive']:
                score += 15 * sentiment_confidence
            elif sentiment.get('label') in ['neutral']:
                score += 10 * sentiment_confidence
            else:
                score += 5 * sentiment_confidence
            
            # Readability bonus (0-10 points)
            score += readability * 10
            
            # Content structure bonus (0-10 points)
            if title and description and keywords:
                score += 10  # Complete content
            elif title and description:
                score += 7
            elif title:
                score += 3
            
            return min(score / 100, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5
    
    def predict_engagement(self, title: str, description: str, 
                          sentiment: Dict[str, Any], quality_score: float,
                          content_type: str, keywords: List[Dict[str, Any]]) -> float:
        """Advanced engagement prediction using multiple signals"""
        try:
            engagement = 0.0
            
            # Title engagement factors (0-30%)
            if title:
                title_lower = title.lower()
                
                # Question and exclamation marks
                engagement += min(title.count('?') * 0.05, 0.1)
                engagement += min(title.count('!') * 0.03, 0.08)
                
                # Engaging words and phrases
                engaging_patterns = [
                    ('amazing', 0.04), ('incredible', 0.04), ('shocking', 0.05),
                    ('secret', 0.03), ('revealed', 0.03), ('ultimate', 0.03),
                    ('best', 0.02), ('worst', 0.02), ('top', 0.02),
                    ('how to', 0.06), ('diy', 0.04), ('tutorial', 0.04)
                ]
                
                for pattern, weight in engaging_patterns:
                    if pattern in title_lower:
                        engagement += weight
                
                # Title length optimization
                title_len = len(title)
                if 40 <= title_len <= 60:  # Optimal for engagement
                    engagement += 0.05
            
            # Content type engagement multiplier (0-20%)
            content_type_multipliers = {
                ContentType.TUTORIAL.value: 0.15,
                ContentType.REVIEW.value: 0.12,
                ContentType.ENTERTAINMENT.value: 0.18,
                ContentType.GAMING.value: 0.16,
                ContentType.MUSIC.value: 0.14,
                ContentType.COMEDY.value: 0.20,
                ContentType.NEWS.value: 0.10,
                ContentType.EDUCATIONAL.value: 0.08
            }
            
            engagement += content_type_multipliers.get(content_type, 0.05)
            
            # Sentiment impact (0-20%)
            sentiment_impact = abs(sentiment.get('polarity', 0)) * 0.15
            if sentiment.get('is_emotional', False):
                sentiment_impact *= 1.5
            engagement += min(sentiment_impact, 0.2)
            
            # Quality score impact (0-25%)
            engagement += quality_score * 0.25
            
            # Keyword diversity bonus (0-15%)
            if keywords:
                unique_categories = set(kw.get('category', 'unknown') for kw in keywords)
                diversity_bonus = min(len(unique_categories) * 0.03, 0.15)
                engagement += diversity_bonus
            
            # Description engagement (0-10%)
            if description:
                desc_len = len(description)
                if 200 <= desc_len <= 500:  # Optimal description length
                    engagement += 0.1
                elif 100 <= desc_len <= 800:
                    engagement += 0.05
            
            return min(engagement, 1.0)
            
        except Exception as e:
            logger.warning(f"Engagement prediction failed: {e}")
            return 0.5
    
    def extract_topics(self, keywords: List[Dict[str, Any]], 
                      content_type: str, text: str) -> List[Dict[str, Any]]:
        """Extract semantic topics with confidence scores"""
        if not keywords:
            return []
        
        try:
            # Define topic clusters
            topic_clusters = {
                'technology': ['tech', 'software', 'app', 'digital', 'computer', 'ai', 'data', 'coding', 'programming'],
                'business': ['business', 'marketing', 'sales', 'finance', 'entrepreneur', 'startup', 'money', 'investment'],
                'entertainment': ['movie', 'music', 'game', 'fun', 'comedy', 'show', 'celebrity', 'entertainment'],
                'education': ['learn', 'education', 'tutorial', 'guide', 'course', 'teaching', 'study', 'knowledge'],
                'health': ['health', 'fitness', 'medical', 'wellness', 'exercise', 'nutrition', 'workout', 'diet'],
                'lifestyle': ['lifestyle', 'travel', 'food', 'fashion', 'home', 'beauty', 'cooking', 'recipe'],
                'sports': ['sport', 'football', 'basketball', 'soccer', 'tennis', 'golf', 'athletic', 'competition'],
                'science': ['science', 'research', 'experiment', 'discovery', 'physics', 'chemistry', 'biology'],
                'arts': ['art', 'design', 'creative', 'painting', 'drawing', 'music', 'photography', 'culture']
            }
            
            # Score topics based on keywords
            topic_scores = {}
            keyword_texts = [kw['text'].lower() for kw in keywords]
            
            for topic, topic_keywords in topic_clusters.items():
                score = 0.0
                matches = 0
                
                for topic_kw in topic_keywords:
                    for user_kw in keyword_texts:
                        if topic_kw in user_kw or user_kw in topic_kw:
                            # Weight by keyword score
                            kw_obj = next((k for k in keywords if k['text'].lower() == user_kw), None)
                            kw_score = kw_obj.get('score', 0.5) if kw_obj else 0.5
                            score += kw_score
                            matches += 1
                
                if matches > 0:
                    # Normalize by number of matches and apply content type boost
                    normalized_score = score / len(topic_keywords)
                    
                    # Content type relevance boost
                    if content_type == ContentType.TECHNOLOGY.value and topic == 'technology':
                        normalized_score *= 1.5
                    elif content_type == ContentType.EDUCATIONAL.value and topic == 'education':
                        normalized_score *= 1.5
                    elif content_type == ContentType.HEALTH_FITNESS.value and topic == 'health':
                        normalized_score *= 1.5
                    
                    topic_scores[topic] = normalized_score
            
            # Sort and format topics
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            
            topics = []
            for topic, score in sorted_topics[:self.config.max_topics]:
                if score > 0.1:  # Minimum confidence threshold
                    topics.append({
                        'name': topic,
                        'confidence': min(score, 1.0),
                        'relevance': 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'
                    })
            
            return topics
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []

class EnrichmentService:
    """
    Production-level video enrichment service with comprehensive features:
    - Advanced NLP analysis
    - Caching and performance optimization
    - Error handling and recovery
    - Monitoring and metrics
    - Batch processing
    - Rate limiting
    """
    
    def __init__(self, 
                 db_session: AsyncSession, 
                 redis_client: Optional[redis.Redis] = None,
                 config: Optional[EnrichmentConfig] = None):
        self.db = db_session
        self.config = config or EnrichmentConfig()
        self.cache_manager = CacheManager(redis_client, self.config)
        self.nlp_processor = AdvancedNLPProcessor(self.config)
        
        # Rate limiting
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._request_count = {}
        self._request_timestamps = {}
        
        # Processing tracking
        self._active_requests: Set[str] = set()
        
        # Initialize metrics server if not already started
        try:
            start_http_server(8000)
            logger.info("Metrics server started on port 8000")
        except Exception:
            pass  # Server might already be running
    
    async def enrich_video(self, 
                          video_id: int, 
                          force_refresh: bool = False,
                          priority: ProcessingPriority = ProcessingPriority.NORMAL,
                          request_id: Optional[str] = None) -> EnrichmentResult:
        """
        Main enrichment pipeline with comprehensive error handling and monitoring
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.time()
        
        # Rate limiting check
        if not await self._check_rate_limit(request_id):
            ENRICHMENT_REQUESTS.labels(status='rate_limited').inc()
            raise EnrichmentError("Rate limit exceeded", status=EnrichmentStatus.RATE_LIMITED)
        
        async with self._request_semaphore:
            ACTIVE_ENRICHMENTS.inc()
            self._active_requests.add(request_id)
            
            try:
                with ENRICHMENT_DURATION.time():
                    logger.info(f"Starting enrichment for video_id: {video_id}, request_id: {request_id}")
                    
                    # 1. Fetch and validate video data
                    video = await self._get_video_with_validation(video_id)
                    content = self._prepare_content(video)
                    
                    # 2. Check cache if not forcing refresh
                    if not force_refresh:
                        cached_result = await self.cache_manager.get_cached_result(video_id, content)
                        if cached_result:
                            logger.info(f"Using cached result for video_id: {video_id}")
                            cached_result.metadata['status'] = EnrichmentStatus.CACHED.value
                            cached_result.request_id = request_id
                            ENRICHMENT_REQUESTS.labels(status='cached').inc()
                            return cached_result
                    
                    # 3. Run enrichment pipeline with timeout
                    result = await asyncio.wait_for(
                        self._run_enrichment_pipeline(video_id, video, content, request_id),
                        timeout=self.config.timeout_seconds
                    )
                    
                    # 4. Finalize result
                    processing_time = time.time() - start_time
                    result.processing_time = processing_time
                    result.metadata['status'] = EnrichmentStatus.COMPLETED.value
                    result.metadata['processed_at'] = datetime.now().isoformat()
                    result.metadata['priority'] = priority.name
                    
                    # 5. Save to database
                    await self._save_enrichment_result(result)
                    
                    # 6. Cache the result
                    await self.cache_manager.cache_result(video_id, content, result)
                    
                    logger.info(f"Successfully completed enrichment for video_id: {video_id} in {processing_time:.2f}s")
                    ENRICHMENT_REQUESTS.labels(status='success').inc()
                    return result
                    
            except asyncio.TimeoutError:
                logger.error(f"Enrichment timeout for video_id: {video_id}")
                ENRICHMENT_REQUESTS.labels(status='timeout').inc()
                return self._create_error_result(video_id, "Processing timeout", time.time() - start_time, request_id)
                
            except Exception as e:
                logger.error(f"Enrichment failed for video_id: {video_id}: {e}", exc_info=True)
                ENRICHMENT_REQUESTS.labels(status='error').inc()
                return self._create_error_result(video_id, str(e), time.time() - start_time, request_id)
                
            finally:
                ACTIVE_ENRICHMENTS.dec()
                self._active_requests.discard(request_id)
    
    async def _check_rate_limit(self, request_id: str) -> bool:
        """Check rate limiting (simple implementation)"""
        current_time = time.time()
        
        # Clean old timestamps
        cutoff_time = current_time - 3600  # 1 hour window
        self._request_timestamps = {
            req_id: timestamp for req_id, timestamp in self._request_timestamps.items()
            if timestamp > cutoff_time
        }
        
        # Check rate limit (100 requests per hour per service instance)
        if len(self._request_timestamps) >= 100:
            return False
        
        self._request_timestamps[request_id] = current_time
        return True
    
    async def _get_video_with_validation(self, video_id: int) -> Video:
        """Fetch and validate video with enhanced error handling"""
        try:
            result = await self.db.execute(select(Video).where(Video.id == video_id))
            video = result.scalar_one_or_none()
            
            if not video:
                raise ValidationError(f"Video with id {video_id} not found")
            
            # Additional validation
            if not video.title and not video.description:
                raise ValidationError(f"Video {video_id} has no content to enrich")
            
            return video
            
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching video {video_id}: {e}")
            raise EnrichmentError(f"Database error: {str(e)}")
    
    def _prepare_content(self, video: Video) -> str:
        """Prepare and clean content for analysis"""
        parts = []
        
        # Add title with higher weight
        if video.title:
            title_clean = video.title.strip()
            if title_clean:
                parts.append(f"TITLE: {title_clean}")
        
        # Add description
        if video.description:
            desc_clean = video.description.strip()
            if desc_clean:
                parts.append(f"DESCRIPTION: {desc_clean}")
        
        # Add metadata if available
        if hasattr(video, 'tags') and video.tags:
            tags = ', '.join(tag.strip() for tag in video.tags if tag.strip())
            if tags:
                parts.append(f"TAGS: {tags}")
        
        if hasattr(video, 'category') and video.category:
            parts.append(f"CATEGORY: {video.category}")
        
        content = "\n\n".join(parts)
        
        # Limit content length
        if len(content) > self.config.max_text_length:
            content = content[:self.config.max_text_length] + "..."
        
        return content
    
    async def _run_enrichment_pipeline(self, 
                                     video_id: int, 
                                     video: Video, 
                                     content: str,
                                     request_id: str) -> EnrichmentResult:
        """Run the complete enrichment pipeline"""
        
        # Language detection
        language, lang_confidence = self.nlp_processor.detect_language(content)
        
        # Sentiment analysis
        sentiment = self.nlp_processor.analyze_sentiment_advanced(content)
        
        # Keyword extraction
        keywords, keyword_scores = self.nlp_processor.extract_keywords_advanced(content)
        
        # Content type classification
        content_type, type_confidence = self.nlp_processor.content_classifier.classify(
            video.title or "", video.description or "", [kw['text'] for kw in keywords]
        )
        
        # Readability analysis
        readability_score = self.nlp_processor.calculate_readability_score(content)
        
        # Quality score calculation
        quality_score = self.nlp_processor.calculate_quality_score(
            video.title or "", video.description or "", keywords, sentiment, readability_score
        )
        
        # Engagement prediction
        engagement_prediction = self.nlp_processor.predict_engagement(
            video.title or "", video.description or "", sentiment, quality_score,
            content_type.value, keywords
        )
        
        # Generate embedding asynchronously
        embedding = await self._generate_embedding_async(content)
        
        # Topic extraction
        topics = self.nlp_processor.extract_topics(keywords, content_type.value, content)
        
        # Confidence scores
        confidence_scores = {
            'language': lang_confidence,
            'sentiment': sentiment.get('confidence', 0.0),
            'content_type': type_confidence,
            'keywords': sum(kw.get('score', 0) for kw in keywords) / max(len(keywords), 1),
            'overall': (lang_confidence + sentiment.get('confidence', 0.0) + type_confidence) / 3
        }
        
        # Enhanced metadata
        metadata = {
            'model_versions': {
                'langdetect': '1.0.9',
                'textblob': '0.17.1',
                'sentence_transformers': '2.2.2',
                'yake': '0.4.8',
                'spacy': '3.4.0'
            },
            'processing_info': {
                'content_length': len(content),
                'keyword_count': len(keywords),
                'topic_count': len(topics),
                'has_title': bool(video.title),
                'has_description': bool(video.description),
                'language_detected': language != 'unknown'
            },
            'quality_metrics': {
                'readability_score': readability_score,
                'sentiment_strength': sentiment.get('intensity', 0.0),
                'keyword_diversity': len(set(kw.get('category', 'unknown') for kw in keywords))
            }
        }
        
        return EnrichmentResult(
            video_id=video_id,
            request_id=request_id,
            language=language,
            language_confidence=lang_confidence,
            sentiment=sentiment,
            keywords=keywords,
            topics=topics,
            content_type=content_type.value,
            content_type_confidence=type_confidence,
            quality_score=quality_score,
            engagement_prediction=engagement_prediction,
            readability_score=readability_score,
            embedding=embedding,
            processing_time=0.0,  # Will be set later
            confidence_scores=confidence_scores,
            metadata=metadata,
            created_at=datetime.now()
        )
    
    async def _generate_embedding_async(self, text: str) -> List[float]:
        """Generate text embedding with error handling and optimization"""
        try:
            embedding_model = model_manager.get_model('embedding_model')
            
            # Truncate text for embedding
            text_for_embedding = text[:self.config.embedding_max_length]
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.nlp_processor.thread_pool,
                lambda: embedding_model.encode([text_for_embedding], show_progress_bar=False)[0]
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            # Return zero vector of standard dimensionality
            return [0.0] * 384  # all-MiniLM-L6-v2 dimension
    
    async def _save_enrichment_result(self, result: EnrichmentResult):
        """Save enrichment result to database with error handling"""
        try:
            # Check if enrichment already exists
            existing = await self.db.execute(
                select(VideoEnrichment).where(VideoEnrichment.video_id == result.video_id)
            )
            existing_enrichment = existing.scalar_one_or_none()
            
            if existing_enrichment:
                # Update existing record
                existing_enrichment.sentiment = result.sentiment['label']
                existing_enrichment.topics = [topic['name'] for topic in result.topics]
                existing_enrichment.embedding = result.embedding
                existing_enrichment.language = result.language
                existing_enrichment.keywords = [kw['text'] for kw in result.keywords]
                existing_enrichment.content_type = result.content_type
                existing_enrichment.quality_score = result.quality_score
                existing_enrichment.engagement_prediction = result.engagement_prediction
                existing_enrichment.confidence_scores = result.confidence_scores
                existing_enrichment.enrichment_metadata = result.metadata
                existing_enrichment.updated_at = datetime.now()
                
                logger.info(f"Updated existing enrichment for video {result.video_id}")
            else:
                # Create new record
                enrichment_data = VideoEnrichment(
                    video_id=result.video_id,
                    sentiment=result.sentiment['label'],
                    topics=[topic['name'] for topic in result.topics],
                    embedding=result.embedding,
                    language=result.language,
                    keywords=[kw['text'] for kw in result.keywords],
                    content_type=result.content_type,
                    quality_score=result.quality_score,
                    engagement_prediction=result.engagement_prediction,
                    confidence_scores=result.confidence_scores,
                    enrichment_metadata=result.metadata
                )
                
                self.db.add(enrichment_data)
                logger.info(f"Created new enrichment for video {result.video_id}")
            
            await self.db.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to save enrichment result: {e}")
            await self.db.rollback()
            raise EnrichmentError(f"Database save failed: {str(e)}")
    
    def _create_error_result(self, video_id: int, error_message: str, 
                           processing_time: float, request_id: str) -> EnrichmentResult:
        """Create a standardized error result"""
        return EnrichmentResult(
            video_id=video_id,
            request_id=request_id,
            language="unknown",
            language_confidence=0.0,
            sentiment={"label": "neutral", "polarity": 0.0, "confidence": 0.0},
            keywords=[],
            topics=[],
            content_type=ContentType.OTHER.value,
            content_type_confidence=0.0,
            quality_score=0.0,
            engagement_prediction=0.0,
            readability_score=0.0,
            embedding=[],
            processing_time=processing_time,
            confidence_scores={"overall": 0.0},
            metadata={
                "status": EnrichmentStatus.FAILED.value,
                "error": error_message,
                "processed_at": datetime.now().isoformat(),
                "error_type": type(EnrichmentError).__name__
            },
            created_at=datetime.now()
        )
    
    async def batch_enrich_videos(self, 
                                video_ids: List[int], 
                                max_concurrent: Optional[int] = None,
                                priority: ProcessingPriority = ProcessingPriority.NORMAL) -> List[EnrichmentResult]:
        """Enrich multiple videos with optimized batch processing"""
        if not video_ids:
            return []
        
        max_concurrent = max_concurrent or min(self.config.max_concurrent_requests, len(video_ids))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def enrich_with_semaphore(video_id: int):
            async with semaphore:
                return await self.enrich_video(video_id, priority=priority)
        
        logger.info(f"Starting batch enrichment for {len(video_ids)} videos")
        
        # Process in batches to avoid overwhelming the system
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i + batch_size]
            tasks = [enrich_with_semaphore(video_id) for video_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_result = self._create_error_result(
                        batch[j], str(result), 0.0, str(uuid.uuid4())
                    )
                    results.append(error_result)
                else:
                    results.append(result)
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(video_ids) + batch_size - 1)//batch_size}")
        
        return results
    
    async def get_enrichment_status(self, video_id: int) -> Optional[Dict[str, Any]]:
        """Get current enrichment status for a video"""
        try:
            result = await self.db.execute(
                select(VideoEnrichment).where(VideoEnrichment.video_id == video_id)
            )
            enrichment = result.scalar_one_or_none()
            
            if enrichment:
                return {
                    'video_id': video_id,
                    'status': 'completed',
                    'language': enrichment.language,
                    'content_type': enrichment.content_type,
                    'quality_score': enrichment.quality_score,
                    'last_updated': enrichment.updated_at.isoformat() if enrichment.updated_at else None,
                    'confidence_scores': enrichment.confidence_scores
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get enrichment status for video {video_id}: {e}")
            return None
    
    async def invalidate_enrichment(self, video_id: int):
        """Invalidate enrichment data for a video"""
        try:
            # Remove from cache
            await self.cache_manager.invalidate_cache(video_id)
            
            # Remove from database
            await self.db.execute(
                VideoEnrichment.__table__.delete().where(VideoEnrichment.video_id == video_id)
            )
            await self.db.commit()
            
            logger.info(f"Invalidated enrichment for video {video_id}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate enrichment for video {video_id}: {e}")
            await self.db.rollback()
            raise
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health status"""
        try:
            model_health = model_manager.get_model_health()
            cache_stats = await self.cache_manager.get_cache_stats()
            
            return {
                'status': 'healthy',
                'active_requests': len(self._active_requests),
                'models': model_health,
                'cache': cache_stats,
                'config': {
                    'max_concurrent_requests': self.config.max_concurrent_requests,
                    'timeout_seconds': self.config.timeout_seconds,
                    'cache_ttl_seconds': self.config.cache_ttl_seconds
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Service initialization and utilities

async def create_enrichment_service(
    db_session: AsyncSession,
    redis_url: Optional[str] = None,
    config: Optional[EnrichmentConfig] = None
) -> EnrichmentService:
    """Factory function to create and initialize enrichment service"""
    
    # Initialize Redis client if URL provided
    redis_client = None
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            await redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    
    # Create service
    service = EnrichmentService(db_session, redis_client, config)
    
    # Preload models in background
    asyncio.create_task(model_manager.preload_all_models())
    
    return service

async def initialize_models():
    """Initialize all models for better startup performance"""
    try:
        await model_manager.preload_all_models()
        logger.info("All NLP models preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")

# Context manager for service lifecycle management
@asynccontextmanager
async def enrichment_service_context(
    db_session: AsyncSession,
    redis_url: Optional[str] = None,
    config: Optional[EnrichmentConfig] = None
):
    """Context manager for enrichment service lifecycle"""
    service = None
    try:
        # Create and initialize service
        service = await create_enrichment_service(db_session, redis_url, config)
        yield service
    except Exception as e:
        logger.error(f"Failed to create enrichment service: {e}")
        raise
    finally:
        # Cleanup resources
        if service and hasattr(service, 'nlp_processor'):
            if hasattr(service.nlp_processor, 'thread_pool'):
                service.nlp_processor.thread_pool.shutdown(wait=True)
        logger.info("Enrichment service cleanup completed")

# Utility functions for service management

def get_default_config() -> EnrichmentConfig:
    """Get default configuration with environment variable overrides"""
    import os
    
    return EnrichmentConfig(
        max_text_length=int(os.getenv('ENRICHMENT_MAX_TEXT_LENGTH', '10000')),
        max_keywords=int(os.getenv('ENRICHMENT_MAX_KEYWORDS', '15')),
        max_topics=int(os.getenv('ENRICHMENT_MAX_TOPICS', '8')),
        embedding_max_length=int(os.getenv('ENRICHMENT_EMBEDDING_MAX_LENGTH', '4096')),
        sentiment_confidence_threshold=float(os.getenv('ENRICHMENT_SENTIMENT_THRESHOLD', '0.3')),
        language_confidence_threshold=float(os.getenv('ENRICHMENT_LANGUAGE_THRESHOLD', '0.5')),
        cache_ttl_seconds=int(os.getenv('ENRICHMENT_CACHE_TTL', str(86400 * 7))),
        timeout_seconds=int(os.getenv('ENRICHMENT_TIMEOUT', '300')),
        retry_attempts=int(os.getenv('ENRICHMENT_RETRY_ATTEMPTS', '3')),
        batch_size=int(os.getenv('ENRICHMENT_BATCH_SIZE', '10')),
        max_concurrent_requests=int(os.getenv('ENRICHMENT_MAX_CONCURRENT', '20'))
    )

async def validate_service_dependencies() -> Dict[str, bool]:
    """Validate that all service dependencies are available"""
    dependencies = {
        'spacy_model': False,
        'sentence_transformers': False,
        'yake': False,
        'textblob': False,
        'langdetect': False
    }
    
    try:
        # Test spaCy model
        import spacy
        nlp = spacy.load("en_core_web_sm")
        dependencies['spacy_model'] = True
    except Exception:
        logger.warning("spaCy model 'en_core_web_sm' not available")
    
    try:
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        dependencies['sentence_transformers'] = True
    except Exception:
        logger.warning("SentenceTransformer model not available")
    
    try:
        # Test YAKE
        import yake
        extractor = yake.KeywordExtractor()
        dependencies['yake'] = True
    except Exception:
        logger.warning("YAKE library not available")
    
    try:
        # Test TextBlob
        from textblob import TextBlob
        blob = TextBlob("test")
        dependencies['textblob'] = True
    except Exception:
        logger.warning("TextBlob library not available")
    
    try:
        # Test langdetect
        from langdetect import detect
        detect("This is a test")
        dependencies['langdetect'] = True
    except Exception:
        logger.warning("langdetect library not available")
    
    return dependencies

# Performance monitoring utilities

class PerformanceMonitor:
    """Monitor and track service performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'cache_hit_rate': 0.0,
            'model_load_times': {}
        }
        self.processing_times = []
        self.cache_hits = 0
        self.cache_requests = 0
    
    def record_request(self, success: bool, processing_time: float, from_cache: bool = False):
        """Record a processing request"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Track processing times
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 1000:  # Keep only last 1000 requests
            self.processing_times = self.processing_times[-1000:]
        
        self.metrics['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
        
        # Track cache performance
        self.cache_requests += 1
        if from_cache:
            self.cache_hits += 1
        
        self.metrics['cache_hit_rate'] = self.cache_hits / self.cache_requests if self.cache_requests > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.metrics,
            'success_rate': self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1),
            'percentile_95': np.percentile(self.processing_times, 95) if self.processing_times else 0.0,
            'percentile_99': np.percentile(self.processing_times, 99) if self.processing_times else 0.0
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Content analysis utilities

class ContentAnalyzer:
    """Additional content analysis utilities"""
    
    @staticmethod
    def extract_hashtags(text: str) -> List[str]:
        """Extract hashtags from text"""
        import re
        hashtags = re.findall(r'#\w+', text.lower())
        return [tag[1:] for tag in hashtags]  # Remove # symbol
    
    @staticmethod
    def extract_mentions(text: str) -> List[str]:
        """Extract mentions from text"""
        import re
        mentions = re.findall(r'@\w+', text.lower())
        return [mention[1:] for mention in mentions]  # Remove @ symbol
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        return urls
    
    @staticmethod
    def calculate_text_complexity(text: str) -> Dict[str, float]:
        """Calculate various text complexity metrics"""
        if not text:
            return {'complexity_score': 0.0, 'avg_word_length': 0.0, 'sentence_variety': 0.0}
        
        words = text.split()
        sentences = text.split('.')
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence length variety
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        sentence_variety = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Overall complexity score
        complexity_score = (avg_word_length / 10) + (sentence_variety / 20)
        complexity_score = min(complexity_score, 1.0)
        
        return {
            'complexity_score': complexity_score,
            'avg_word_length': avg_word_length,
            'sentence_variety': sentence_variety,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0
        }

# Service configuration validation

def validate_enrichment_config(config: EnrichmentConfig) -> List[str]:
    """Validate enrichment configuration and return list of issues"""
    issues = []
    
    if config.max_text_length < 100:
        issues.append("max_text_length should be at least 100 characters")
    
    if config.max_keywords < 1:
        issues.append("max_keywords should be at least 1")
    
    if config.max_topics < 1:
        issues.append("max_topics should be at least 1")
    
    if config.timeout_seconds < 30:
        issues.append("timeout_seconds should be at least 30 seconds")
    
    if config.max_concurrent_requests < 1:
        issues.append("max_concurrent_requests should be at least 1")
    
    if config.cache_ttl_seconds < 3600:
        issues.append("cache_ttl_seconds should be at least 1 hour for efficiency")
    
    return issues

# Export all public classes and functions
__all__ = [
    # Main service class
    'EnrichmentService',
    
    # Configuration and data classes
    'EnrichmentConfig',
    'EnrichmentResult',
    'EnrichmentStatus',
    'ContentType',
    'ProcessingPriority',
    
    # Utility classes
    'ModelManager',
    'CacheManager',
    'AdvancedNLPProcessor',
    'ContentTypeClassifier',
    'ContentAnalyzer',
    'PerformanceMonitor',
    
    # Exception classes
    'EnrichmentError',
    'ModelLoadError',
    'ValidationError',
    
    # Factory and utility functions
    'create_enrichment_service',
    'enrichment_service_context',
    'initialize_models',
    'get_default_config',
    'validate_service_dependencies',
    'validate_enrichment_config',
    
    # Global instances
    'model_manager',
    'performance_monitor'
]