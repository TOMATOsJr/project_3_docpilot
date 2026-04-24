# Context Budgeting & Semantic Search Walkthrough

The application has been successfully updated to incorporate true vector embeddings via `litellm`, rigorous 90% context budget enforcement, and the strict priority allocation strategy you requested. 

## What Was Changed

### 1. Vector Embedding on Ingestion
When documents are uploaded and chunked, `backend/app/services/ingest_service.py` now routes each chunk through `litellm.embedding()`.
- **Default Model:** `text-embedding-3-small` (Configurable via `.env` with `EMBEDDING_MODEL`).
- Embeddings are persisted via the PostgreSQL repository alongside their parent chunks.
> [!IMPORTANT]
> Because embeddings are generated at upload time, **any documents uploaded prior to this update will not have embeddings**. You will need to delete and re-upload your existing documents from the UI so they can be processed by the embedding model.

### 2. In-Memory Cosine Similarity 
To avoid requiring a destructive database migration or enforcing `pgvector` container dependencies right away, the vector search logic was built natively.
- `PostgresRepository.search_chunks_by_embedding` computes cosine similarity between your query's vector and the document vectors.
- **Strict Cutoff:** Any chunk with a similarity score `<= 0.6` is immediately discarded. The remaining chunks are sorted in descending order of relevance.

### 3. Context Budget Allocation (100% Chunks -> Leftover History)
The RAG Engine's prompt constructor (`_build_grounded_prompt`) was entirely rewritten into an orchestrator method: `_apply_context_budget`. 

Here is exactly how the 90% budget limit is calculated and filled:
1. It queries `litellm` for the absolute `max_input_tokens` of the currently selected LLM (e.g., Gemini 1.5 Flash).
2. It calculates the 90% budget.
3. It subtracts the base system instructions and the user's query using `litellm.token_counter`.
4. **Chunks First:** It iterates through the sorted, `> 0.6` similarity chunks. It counts the tokens of each chunk and adds them to the context window until the budget runs out.
5. **History Last:** If (and only if) all valid chunks have been added and there is still room left in the budget, it will iterate backward through your conversation history, adding previous turns until the remainder of the budget is filled. 

## Verification

You can verify the behavior by doing the following:
1. Reload the frontend and re-upload `idea.md` or a PDF document.
2. Ask a question. You will notice a slight delay compared to before; this is the system generating the query vector and running the cosine similarity calculation.
3. The LLM will now provide significantly more accurate citations, as it is no longer relying on simple keyword overlap.
