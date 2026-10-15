# Runtime map: old custom pipeline → LangChain components

This document is a learning companion for the rewritten runtime.

1. `src/ingestion/document_store.py` calls `PyPDFLoader` and `RecursiveCharacterTextSplitter`. The JSON chunk registry remains application-specific persistence metadata; the loader/splitter themselves are LangChain integrations.
2. `src/langchain_rag.py:create_embeddings` returns `HuggingFaceEmbeddings`. `PersistentFaissStore` converts the stored chunk records into LangChain `Document` objects and creates/loads `langchain_community.vectorstores.FAISS` collections.
3. `build_conversational_rag_chain` first prepares memory, then uses `RunnableBranch` to decide whether the latest question needs rewriting. The rewrite chain is an LCEL expression: mapping lambda → `ChatPromptTemplate` → chat model → `StrOutputParser`.
4. A retriever callable queries the FAISS retriever. In hybrid mode it also invokes `BM25Retriever`, then fuses ranked lists with reciprocal rank fusion. The optional CrossEncoder orders the candidate documents after retrieval.
5. Evidence is formatted with source filename/page metadata. A token budget limits the context sent to the answer prompt. The answer stage is another LCEL chain.
6. `RunnableWithMessageHistory` reads and writes messages through `SQLiteChatMessageHistory`. Full history remains in SQLite; only selected recent messages and the rolling summary are passed to the model. Older messages are summarized by a separate LCEL chain.
7. `app.py` stays thin: collect uploads and user input, call the service, render the answer/sources/diagnostics.

The historical `src/rag_pipeline.py` remains so earlier experiments can still be reproduced. The Streamlit runtime does not import it.
