# app/domains/search/service.py
import re
import math
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from enum import Enum

from rapidfuzz import fuzz, process
import unicodedata

from app.domains.search.schemas import CreatorProfile, SearchResult, SearchType


class ScoreWeight(Enum):
    """Weights for different matching criteria."""
    EXACT_NAME_MATCH = 10.0
    EXACT_HANDLE_MATCH = 8.0
    PARTIAL_NAME_MATCH = 6.0
    PARTIAL_HANDLE_MATCH = 5.0
    BIO_KEYWORD_MATCH = 2.0
    FOLLOWER_BOOST = 1.5
    VERIFIED_BOOST = 1.2
    ENGAGEMENT_BOOST = 1.3


@dataclass
class SearchContext:
    """Context for search operations with preprocessing results."""
    query: str
    normalized_query: str
    keywords: Set[str]
    query_tokens: List[str]
    search_type: SearchType
    
    
@dataclass
class ProfileScore:
    """Detailed scoring breakdown for a profile."""
    profile: CreatorProfile
    base_score: float
    name_similarity: float
    handle_similarity: float
    keyword_matches: int
    social_signals: float
    final_score: float
    match_reasons: List[str]


class TextProcessor:
    """Advanced text processing utilities."""
    
    # Common stop words that don't contribute to relevance
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    }
    
    # Common social media terms
    SOCIAL_TERMS = {
        'official', 'verified', 'channel', 'page', 'account', 'profile',
        'creator', 'influencer', 'youtuber', 'content', 'subscribe'
    }
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for better matching."""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove accents and diacritics
        text = ''.join(c for c in text if not unicodedata.combining(c))
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @classmethod
    def extract_keywords(cls, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        normalized = cls.normalize_text(text)
        tokens = normalized.split()
        
        # Filter out stop words and very short tokens
        keywords = {
            token for token in tokens 
            if len(token) >= 2 and token not in cls.STOP_WORDS
        }
        
        return keywords
    
    @classmethod
    def get_important_terms(cls, text: str) -> Set[str]:
        """Extract the most important terms from text."""
        keywords = cls.extract_keywords(text)
        
        # Prioritize longer terms and non-social terms
        important = set()
        for keyword in keywords:
            if len(keyword) >= 3 and keyword not in cls.SOCIAL_TERMS:
                important.add(keyword)
        
        return important if important else keywords


class SearchService:
    """
    Enhanced search service with sophisticated scoring algorithms,
    better text processing, and comprehensive relevance calculation.
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def score_results(
        self, query: str, profiles: List[CreatorProfile], search_type: SearchType
    ) -> List[SearchResult]:
        """
        Routes to the appropriate scoring method based on the search type.
        """
        if not query.strip() or not profiles:
            return []
        
        context = self._create_search_context(query, search_type)
        
        if search_type == SearchType.CREATOR:
            return self._score_by_name_similarity(context, profiles)
        elif search_type == SearchType.TOPIC:
            return self._score_by_keyword_relevance(context, profiles)
        
        return []
    
    def _create_search_context(self, query: str, search_type: SearchType) -> SearchContext:
        """Create a search context with preprocessed query data."""
        normalized_query = self.text_processor.normalize_text(query)
        keywords = self.text_processor.extract_keywords(query)
        query_tokens = normalized_query.split()
        
        return SearchContext(
            query=query,
            normalized_query=normalized_query,
            keywords=keywords,
            query_tokens=query_tokens,
            search_type=search_type
        )
    
    def _score_by_name_similarity(
        self, context: SearchContext, profiles: List[CreatorProfile]
    ) -> List[SearchResult]:
        """
        Enhanced name-based scoring with multiple similarity algorithms
        and social signal boosting.
        """
        scored_profiles: List[ProfileScore] = []
        
        for profile in profiles:
            score_details = self._calculate_name_similarity_score(context, profile)
            scored_profiles.append(score_details)
        
        # Sort by final score
        scored_profiles.sort(key=lambda x: x.final_score, reverse=True)
        
        # Convert to SearchResult objects
        return [
            SearchResult(
                profile=score.profile,
                match_confidence=min(score.final_score, 1.0),  # Cap at 1.0
                match_details={
                    'name_similarity': score.name_similarity,
                    'handle_similarity': score.handle_similarity,
                    'social_signals': score.social_signals,
                    'match_reasons': score.match_reasons
                }
            )
            for score in scored_profiles
        ]
    
    def _calculate_name_similarity_score(
        self, context: SearchContext, profile: CreatorProfile
    ) -> ProfileScore:
        """Calculate detailed similarity score for name-based search."""
        match_reasons = []
        
        # Normalize profile data
        norm_name = self.text_processor.normalize_text(profile.name)
        norm_handle = self.text_processor.normalize_text(profile.handle)
        
        # Multiple similarity algorithms
        name_scores = self._calculate_text_similarity(context.normalized_query, norm_name)
        handle_scores = self._calculate_text_similarity(context.normalized_query, norm_handle)
        
        # Base similarity score
        name_similarity = max(name_scores.values())
        handle_similarity = max(handle_scores.values())
        
        base_score = max(name_similarity, handle_similarity)
        
        # Add match reasons
        if name_similarity > 0.8:
            match_reasons.append(f"High name similarity ({name_similarity:.2f})")
        if handle_similarity > 0.8:
            match_reasons.append(f"High handle similarity ({handle_similarity:.2f})")
        
        # Check for exact matches
        if context.normalized_query == norm_name:
            base_score = max(base_score, 0.95)
            match_reasons.append("Exact name match")
        elif context.normalized_query == norm_handle:
            base_score = max(base_score, 0.90)
            match_reasons.append("Exact handle match")
        
        # Social signals boost
        social_signals = self._calculate_social_signals(profile)
        if social_signals > 0.1:
            match_reasons.append(f"Social signals boost ({social_signals:.2f})")
        
        # Final score with social boost
        final_score = base_score * (1 + social_signals)
        
        return ProfileScore(
            profile=profile,
            base_score=base_score,
            name_similarity=name_similarity,
            handle_similarity=handle_similarity,
            keyword_matches=0,
            social_signals=social_signals,
            final_score=final_score,
            match_reasons=match_reasons
        )
    
    def _score_by_keyword_relevance(
        self, context: SearchContext, profiles: List[CreatorProfile]
    ) -> List[SearchResult]:
        """
        Enhanced keyword-based scoring with TF-IDF-like weighting,
        semantic matching, and comprehensive relevance calculation.
        """
        # Calculate term frequencies across all profiles
        term_frequencies = self._calculate_term_frequencies(profiles, context.keywords)
        
        scored_profiles: List[ProfileScore] = []
        
        for profile in profiles:
            score_details = self._calculate_keyword_relevance_score(
                context, profile, term_frequencies
            )
            scored_profiles.append(score_details)
        
        # Filter out profiles with zero relevance
        relevant_profiles = [p for p in scored_profiles if p.final_score > 0]
        
        if not relevant_profiles:
            return []
        
        # Normalize scores
        max_score = max(p.final_score for p in relevant_profiles)
        if max_score > 0:
            for profile_score in relevant_profiles:
                profile_score.final_score = profile_score.final_score / max_score
        
        # Sort by final score
        relevant_profiles.sort(key=lambda x: x.final_score, reverse=True)
        
        return [
            SearchResult(
                profile=score.profile,
                match_confidence=score.final_score,
                match_details={
                    'keyword_matches': score.keyword_matches,
                    'social_signals': score.social_signals,
                    'match_reasons': score.match_reasons
                }
            )
            for score in relevant_profiles
        ]
    
    def _calculate_keyword_relevance_score(
        self, 
        context: SearchContext, 
        profile: CreatorProfile,
        term_frequencies: Dict[str, int]
    ) -> ProfileScore:
        """Calculate detailed keyword relevance score."""
        match_reasons = []
        keyword_matches = 0
        base_score = 0.0
        
        # Get profile text data
        profile_texts = {
            'name': self.text_processor.normalize_text(profile.name),
            'handle': self.text_processor.normalize_text(profile.handle),
            'bio': self.text_processor.normalize_text(profile.bio) if profile.bio else ""
        }
        
        # Calculate keyword matching scores
        for keyword in context.keywords:
            keyword_score = 0.0
            matches_in = []
            
            # Check each field with different weights
            field_weights = {'name': 3.0, 'handle': 2.5, 'bio': 1.0}
            
            for field, text in profile_texts.items():
                if keyword in text:
                    # TF-IDF-like scoring
                    term_frequency = text.count(keyword)
                    inverse_doc_freq = math.log(len(term_frequencies) / (term_frequencies.get(keyword, 1) + 1))
                    
                    field_score = term_frequency * inverse_doc_freq * field_weights[field]
                    keyword_score += field_score
                    matches_in.append(field)
            
            if keyword_score > 0:
                keyword_matches += 1
                base_score += keyword_score
                match_reasons.append(f"'{keyword}' in {', '.join(matches_in)}")
        
        # Phrase matching bonus
        if len(context.query_tokens) > 1:
            phrase_score = self._calculate_phrase_matching(context, profile_texts)
            if phrase_score > 0:
                base_score += phrase_score
                match_reasons.append(f"Phrase matching bonus ({phrase_score:.2f})")
        
        # Semantic similarity bonus
        semantic_score = self._calculate_semantic_similarity(context, profile_texts)
        if semantic_score > 0:
            base_score += semantic_score
            match_reasons.append(f"Semantic similarity ({semantic_score:.2f})")
        
        # Social signals
        social_signals = self._calculate_social_signals(profile)
        if social_signals > 0.1:
            match_reasons.append(f"Social signals boost ({social_signals:.2f})")
        
        # Final score with social boost
        final_score = base_score * (1 + social_signals)
        
        return ProfileScore(
            profile=profile,
            base_score=base_score,
            name_similarity=0.0,
            handle_similarity=0.0,
            keyword_matches=keyword_matches,
            social_signals=social_signals,
            final_score=final_score,
            match_reasons=match_reasons
        )
    
    def _calculate_text_similarity(self, query: str, text: str) -> Dict[str, float]:
        """Calculate multiple text similarity scores."""
        if not query or not text:
            return {'ratio': 0.0, 'partial': 0.0, 'token_sort': 0.0, 'token_set': 0.0}
        
        return {
            'ratio': fuzz.ratio(query, text) / 100.0,
            'partial': fuzz.partial_ratio(query, text) / 100.0,
            'token_sort': fuzz.token_sort_ratio(query, text) / 100.0,
            'token_set': fuzz.token_set_ratio(query, text) / 100.0
        }
    
    def _calculate_term_frequencies(
        self, profiles: List[CreatorProfile], keywords: Set[str]
    ) -> Dict[str, int]:
        """Calculate how many profiles contain each keyword."""
        term_freq = Counter()
        
        for profile in profiles:
            profile_text = " ".join([
                self.text_processor.normalize_text(profile.name),
                self.text_processor.normalize_text(profile.handle),
                self.text_processor.normalize_text(profile.bio) if profile.bio else ""
            ])
            
            profile_keywords = self.text_processor.extract_keywords(profile_text)
            
            for keyword in keywords:
                if keyword in profile_keywords:
                    term_freq[keyword] += 1
        
        return dict(term_freq)
    
    def _calculate_phrase_matching(
        self, context: SearchContext, profile_texts: Dict[str, str]
    ) -> float:
        """Calculate bonus for phrase/sequence matching."""
        if len(context.query_tokens) < 2:
            return 0.0
        
        phrase_score = 0.0
        query_phrase = " ".join(context.query_tokens)
        
        for field, text in profile_texts.items():
            if query_phrase in text:
                # Bonus based on field importance and phrase length
                field_weight = {'name': 2.0, 'handle': 1.5, 'bio': 0.5}[field]
                length_bonus = min(len(context.query_tokens) * 0.2, 1.0)
                phrase_score += field_weight * length_bonus
        
        return phrase_score
    
    def _calculate_semantic_similarity(
        self, context: SearchContext, profile_texts: Dict[str, str]
    ) -> float:
        """Calculate semantic similarity bonus (simplified version)."""
        # This is a simplified version. In production, you might use
        # word embeddings or more sophisticated NLP techniques
        
        semantic_score = 0.0
        
        # Check for related terms and synonyms
        query_terms = self.text_processor.get_important_terms(context.query)
        
        for field, text in profile_texts.items():
            profile_terms = self.text_processor.get_important_terms(text)
            
            # Simple term overlap with stemming-like matching
            for query_term in query_terms:
                for profile_term in profile_terms:
                    # Check for prefix matching (simple stemming)
                    if (len(query_term) >= 3 and len(profile_term) >= 3 and
                        (query_term.startswith(profile_term[:3]) or 
                         profile_term.startswith(query_term[:3]))):
                        
                        field_weight = {'name': 1.0, 'handle': 0.8, 'bio': 0.3}[field]
                        semantic_score += field_weight * 0.5
        
        return min(semantic_score, 2.0)  # Cap semantic bonus
    
    def _calculate_social_signals(self, profile: CreatorProfile) -> float:
        """Calculate social signals boost based on profile metrics."""
        signals = 0.0
        
        # Follower count boost (logarithmic scaling)
        if profile.followers_count and profile.followers_count > 0:
            follower_boost = min(math.log10(profile.followers_count) / 10, 0.3)
            signals += follower_boost
        
        # Verification boost
        if profile.is_verified:
            signals += 0.2
        
        # Engagement metrics (if available in metadata)
        if hasattr(profile, 'metadata') and profile.metadata:
            engagement_rate = profile.metadata.get('engagement_rate', 0)
            if engagement_rate > 0:
                signals += min(engagement_rate / 100, 0.1)
        
        # Video/content count boost
        if profile.video_count and profile.video_count > 0:
            content_boost = min(math.log10(profile.video_count) / 20, 0.1)
            signals += content_boost
        
        return min(signals, 0.6)  # Cap total social signals boost
    
    def get_suggestions(
        self, query: str, profiles: List[CreatorProfile], max_suggestions: int = 5
    ) -> List[str]:
        """Generate search suggestions based on profile data."""
        if not query or not profiles:
            return []
        
        suggestions = set()
        query_lower = query.lower()
        
        # Collect potential suggestions from profile data
        for profile in profiles:
            # Name suggestions
            if profile.name.lower().startswith(query_lower):
                suggestions.add(profile.name)
            
            # Handle suggestions
            if profile.handle.lower().startswith(query_lower):
                suggestions.add(profile.handle)
            
            # Keyword suggestions from bio
            if profile.bio:
                bio_keywords = self.text_processor.extract_keywords(profile.bio)
                for keyword in bio_keywords:
                    if keyword.startswith(query_lower) and len(keyword) > len(query):
                        suggestions.add(keyword.title())
        
        # Sort by length and relevance
        sorted_suggestions = sorted(
            suggestions, 
            key=lambda x: (len(x), x.lower())
        )
        
        return sorted_suggestions[:max_suggestions]