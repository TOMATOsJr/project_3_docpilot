import asyncio
from app.api.deps import get_repository

async def check():
    repo = get_repository()
    docs = repo.list_documents()
    print(f"Total documents: {len(docs)}")
    for doc in docs:
        chunks = repo.get_chunks_by_document(doc.id)
        print(f"Doc {doc.id} has {len(chunks)} chunks.")
        
    session = repo._get_session()
    from app.infrastructure.persistence import ChunkRecord
    records = session.query(ChunkRecord).all()
    count_with_embedding = sum(1 for r in records if r.embedding is not None)
    print(f"Total chunks: {len(records)}, Chunks with embedding: {count_with_embedding}")
    session.close()

if __name__ == "__main__":
    asyncio.run(check())
