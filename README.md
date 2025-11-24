### Day 1 📚 (October 19, 2025)
Foundation: Understanding AI & LLMs

What I Learned:

✅ What are Large Language Models (LLMs)?
✅ How APIs work and why they matter
✅ Basics of prompt engineering
✅ Understanding tokens and model limitations
✅ Setting up development environment

Topics Covered:

Introduction to OpenAI/Gemini APIs
Understanding API keys and authentication
Basic Python setup for AI projects
Reading API documentation
Resources Used:
OpenAI documentation
YouTube tutorials on LLMs
AI prompt engineering guides
Time: 3 hours
Cost: $0.00 (free tier exploration)
Key Takeaway: AI isn't magic—it's patterns + probability!

### Day 2 ✅ (October 20, 2025)
**Built Production Chatbot**

**What I Made:**
- ✅ Chatbot that remembers conversations
- ✅ Streaming responses (types word-by-word!)
- ✅ Error handling (doesn't crash)
- ✅ Save conversations to file
- ✅ 3 versions: basic → streaming → production

**What I Learned:**
- How chatbots remember (conversation history)
- How to make responses stream
- Error handling is important!
- Production code = AI + good software engineering

**Time:** 4 hours
**Cost:** $0.00 (free!)
**Fun Level:** 🔥🔥🔥🔥🔥

### Day 3 📚 (October 21, 2025)
Advanced Concepts: Context & Reasoning

What I Learned:

✅ How context windows work
✅ Chain-of-thought prompting
✅ API rate limits and best practices
✅ Understanding agent architectures
✅ When to use streaming vs batch responses

Topics Covered:

Prompt engineering strategies
ReAct framework (Reasoning + Acting)
Planning complex AI workflows
Error handling patterns in production
Resources Used:
Research papers on AI agents
API documentation deep-dive
Community best practices

Time: 3.5 hours
Cost: $0.00 (research & planning)
Key Takeaway: Good prompts = good results. The prompt is the program!

### Day 4: API Integration and Reasoning Engine
- Added reasoning_engine.py for advanced prompting strategies.
- Documented tests and reflections in REASONING_RESULTS.md.
- Pushed all code and documentation to GitHub.
- Learned how agents solve tasks step by step using APIs.

### Day 5 📚 (November 20, 2025)
RAG Foundations: Vector Databases & Embeddings

What I Learned:

✅ What is RAG (Retrieval-Augmented Generation)?
✅ How embeddings work (text → numbers)
✅ Vector similarity and semantic search
✅ ChromaDB architecture and setup
✅ Document chunking strategies

Topics Covered:

Embeddings and vector spaces
Semantic search vs keyword search
ChromaDB vs other vector databases
Metadata tracking for citations
Chunking strategies (fixed size, semantic)
Resources Used:
ChromaDB documentation
Vector database tutorials
RAG architecture papers

Time: 4 hours
Cost: $0.00 (local setup)
Key Takeaway: RAG solves hallucinations by grounding AI in real documents!

### Day 6 ✅ (November 21, 2025)
**Document Q&A with Citation Tracking - Full RAG System**

**What I Built:**
- Complete RAG (Retrieval-Augmented Generation) pipeline
- Document ingestion with semantic chunking
- Citation-aware prompt engineering
- Q&A system that cites its sources

**Files:**
- `document_ingestion.py` - Load, chunk, embed documents
- `citation_prompts.py` - Prompt templates for citations
- `document_qa.py` - Complete Q&A system
- `qa_comparison.py` - RAG vs direct LLM demo
- `documents/` - Sample ML documents

**Key Achievement:** 
Answers are now grounded in documents, not hallucinations!

**Architecture:**
```
Question → Embed → Search Docs → Build Prompt → Generate → Cite Sources
```
**Example:**
```
Q: "What is machine learning?"
A: "Machine learning is a subset of AI [intro_to_ml.txt] that 
    learns from data without explicit programming [intro_to_ml.txt]."
```
**Time:** 2.5 hours | **Cost:** $0.00 (free tier)

### Week 2, Day 1 ✅ (November 24)
**Automated ETL Data Pipeline**

**Completed:**
- ✅ Built a modular ETL (Extract, Transform, Load) architecture.
- ✅ Implemented robust web scraping with `requests` and `BeautifulSoup`.
- ✅ Added **Resilience Patterns**: Exponential backoff and rate limiting.
- ✅ Developed **Mock Mode** to handle API quotas (`429` errors) gracefully without crashing.
- ✅ Implemented **Semantic Chunking** to prepare text for vector embedding.
- ✅ integrated **ChromaDB** for local vector storage using "Upserts" (Idempotency).
- ✅ Validated system with a full Unit Test suite (`unittest`).

**Key Insight:** "Code that works" isn't enough. Production code must handle failure (like API limits) automatically. Building the "Mock Mode" taught me how to make systems resilient even when external dependencies fail.

**Tech Stack:** Python 3.11, ChromaDB, Google Gemini API, BeautifulSoup4.
**Status:** Component Tested & Ready 🚀


