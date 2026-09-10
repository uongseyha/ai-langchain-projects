def mock_research_data():
    return [
        (
            """
            Attention Mechanisms in Neural Networks

            The attention mechanism was introduced in "Attention Is All You Need"
            by Vaswani et al. (2017). It allows models to focus on relevant parts
            of the input when generating output.

            Key concepts:
            - Query, Key, Value (QKV) triplets
            - Scaled dot-product attention
            - Multi-head attention for parallel processing

            The transformer architecture has become the foundation for modern NLP
            models including BERT, GPT, and T5.
            """,
            "attention_mechanisms.pdf",
        ),
        (
            """
            Retrieval-Augmented Generation (RAG)

            RAG combines retrieval systems with generative models. First introduced
            by Lewis et al. (2020), RAG addresses the limitation of LLMs being
            limited to their training data.

            Components of a RAG system:
            1. Document store with vector embeddings
            2. Retriever to find relevant documents
            3. Generator (LLM) to produce responses

            Benefits include reduced hallucination, up-to-date information,
            and source attribution.
            """,
            "rag_survey.pdf",
        ),
        (
            """
            LangChain and LangGraph Framework Overview

            LangChain is an open-source framework for building LLM applications.
            Key features include modular components, integration with 50+ LLM
            providers, and built-in RAG utilities.

            LangGraph extends LangChain for stateful applications with
            graph-based state management, support for cycles and loops,
            and human-in-the-loop workflows.
            """,
            "langchain_docs.md",
        ),
    ]


def seed_mock_data(assistant):
    for text, source in mock_research_data():
        assistant.add_text(text=text, source=source)
