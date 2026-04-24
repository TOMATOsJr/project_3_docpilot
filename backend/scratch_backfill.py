import asyncio
import litellm
import logging
from app.api.deps import get_repository
from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def backfill():
    repo = get_repository()
    settings = get_settings()
    import os
    os.environ["GEMINI_API_KEY"] = settings.gemini_api_key or ""
    session = repo._get_session()
    
    from app.infrastructure.persistence import ChunkRecord
    records = session.query(ChunkRecord).filter(ChunkRecord.embedding == None).all()
    
    logger.info(f"Found {len(records)} chunks without embeddings. Backfilling using {settings.embedding_model}...")
    
    for i, record in enumerate(records):
        try:
            response = litellm.embedding(model=settings.embedding_model, input=[record.text])
            embedding = response.data[0]['embedding']
            import json
            record.embedding = json.dumps(embedding)
            if i % 10 == 0:
                session.commit()
                logger.info(f"Processed {i+1}/{len(records)}")
        except Exception as e:
            logger.error(f"Failed on chunk {record.id}: {e}")
            
    session.commit()
    logger.info("Backfill complete.")
    session.close()

if __name__ == "__main__":
    asyncio.run(backfill())
