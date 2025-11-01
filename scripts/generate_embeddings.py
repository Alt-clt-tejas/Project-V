# scripts/generate_embeddings.py
import asyncio
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import get_session
from app.database.models import CreatorProfileDB

async def generate_embeddings():
    """Generate embeddings for all creator profiles that don't have them."""
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async for session in get_session():
        # Get all profiles without embeddings
        profiles = await session.execute(
            select(CreatorProfileDB).where(CreatorProfileDB.embedding.is_(None))
        )
        profiles = profiles.scalars().all()
        
        for profile in profiles:
            # Combine relevant text for embedding
            text_to_embed = f"{profile.creator.name} {profile.creator.description}"
            
            # Generate embedding
            embedding = model.encode(text_to_embed)
            
            # Update profile
            profile.embedding = embedding
        
        # Commit changes
        await session.commit()

if __name__ == "__main__":
    asyncio.run(generate_embeddings())