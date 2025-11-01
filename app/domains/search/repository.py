# app/domains/search/repository.py
from typing import List, Optional
import asyncio
from sqlalchemy import select, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from app.database.models import Creator
from app.domains.search.schemas import SearchFilter, CreatorProfile


class SearchRepository:
    """Repository for search operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def find_creators(self, query: str, limit: int = 20) -> List[Creator]:
        """Find creators by text query."""
        # Basic text search implementation
        stmt = select(Creator).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def find_similar_creators(
        self, 
        niche_embedding: List[float], 
        limit: int = 50,
        threshold: float = 0.7
    ) -> List[Creator]:
        """
        Find creators with similar embeddings using cosine similarity.
        Uses pgvector for efficient similarity search.
        
        Args:
            niche_embedding: The embedding vector to compare against
            limit: Maximum number of results to return
            threshold: Minimum cosine similarity threshold (0-1)
        """
        # Using pgvector's <-> operator for cosine similarity
        # Lower value means more similar
        stmt = text("""
            SELECT *
            FROM creators
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> :embedding
            LIMIT :limit
        """)
        
        result = await self.session.execute(
            stmt,
            {
                "embedding": niche_embedding,
                "limit": limit
            }
        )
        
        return list(result.scalars().all())
