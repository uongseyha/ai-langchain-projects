import shutil

from langchain_core.documents import Document
from main import AIResearchAssistant, ResearchResponse
from data import mock_research_data, seed_mock_data


def make_assistant():
    return AIResearchAssistant(persist_directory="./research_db")


def print_research_response(question: str, response: ResearchResponse):
    """Pretty print a structured research response for test output."""

    print(f"\nQ: {question}")
    print(f"\n  Answer: {response.answer}")
    print(f"\n  Confidence: {response.confidence}")
    print(f"  Sources: {', '.join(response.sources)}")

    if response.key_quotes:
        print("\n  Key Quotes:")
        for quote in response.key_quotes:
            print(f'    - "{quote}"')

    print("\n  Follow-up Questions:")
    for follow_up_question in response.follow_up_questions:
        print(f"    - {follow_up_question}")


def run_add_documents_test():
    assistant = make_assistant()
    doc = Document(
        page_content="This is a smoke test for the AI research assistant. It verifies that the assistant can ingest a basic document and split it into chunks.",
        metadata={
            "source": "smoke_test.txt",
            "author": "Copilot",
            "date": "2026-09-10",
        },
    )

    chunk_count = assistant.add_documents([doc], source_name="smoke_test.txt")
    print(f"Smoke test passed: {chunk_count} chunks added")
    assert chunk_count > 0


def run_add_text_test():
    assistant = make_assistant()
    chunk_count = assistant.add_text(
        text="This is a separate smoke test for the add_text method. It verifies that a plain string can be ingested successfully.",
        source="add_text_smoke_test.txt",
        metadata={"author": "Copilot", "date": "2026-09-10"},
    )
    print(f"add_text smoke test passed: {chunk_count} chunks added")
    assert chunk_count > 0


def run_build_retriever_test():
    assistant = make_assistant()
    seed_mock_data(assistant)

    retriever = assistant._build_retriever()
    docs = retriever.invoke("What is attention in neural networks?")

    print(f"_build_retriever smoke test passed: {len(docs)} documents retrieved")
    assert retriever is not None
    assert isinstance(docs, list)
    assert len(docs) > 0


def run_format_docs_for_context_test():
    assistant = make_assistant()
    docs = [
        Document(
            page_content="Retrieval is the process of finding relevant passages from indexed documents.",
            metadata={"source": "retrieval_notes.txt"},
        ),
        Document(
            page_content="Context formatting helps the model answer questions using only the most relevant chunks.",
            metadata={"source": "context_notes.txt"},
        ),
    ]

    formatted = assistant._format_docs_for_context(docs)
    print("_format_docs_for_context smoke test passed:")
    print(formatted)
    assert "[Source 1: retrieval_notes.txt]" in formatted
    assert "[Source 2: context_notes.txt]" in formatted
    assert "Retrieval is the process" in formatted
    assert "Context formatting helps" in formatted


def run_ask_test():
    assistant = make_assistant()
    seed_mock_data(assistant)

    response = assistant.ask("What is attention in neural networks?")

    print("ask() smoke test passed:")
    print(response)
    assert isinstance(response, str)
    assert len(response) > 0
    assert "attention" in response.lower()


def run_ask_failure_test():
    assistant = make_assistant()
    seed_mock_data(assistant)

    response = assistant.ask("What is .NET?")

    print("ask() failure-case smoke test passed:")
    print(response)
    assert isinstance(response, str)
    assert len(response) > 0
    assert any(
        word in response.lower()
        for word in ["not found", "doesn't have", "context", "unable", "no relevant"]
    )


def run_ask_session_test():
    assistant = make_assistant()
    seed_mock_data(assistant)

    q = "What is attention in neural networks?"
    response_a = assistant.ask(q, session_id="session_a")
    response_b = assistant.ask(q, session_id="session_b")

    print("ask() session smoke test passed:")
    print(f"Session A: {response_a}")
    print(f"Session B: {response_b}")
    assert isinstance(response_a, str) and len(response_a) > 0
    assert isinstance(response_b, str) and len(response_b) > 0
    assert "attention" in response_a.lower()
    assert "attention" in response_b.lower()
    assert response_a == response_b


def run_ask_session_scenario_test():
    assistant = make_assistant()
    seed_mock_data(assistant)

    first = assistant.ask("What is attention in neural networks?", session_id="scenario_session")
    second = assistant.ask("What are the key concepts mentioned?", session_id="scenario_session")
    third = assistant.ask("What is the first question I ask?", session_id="scenario_session")

    print("ask() session scenario smoke test passed:")
    print(f"First: {first}")
    print(f"Second: {second}")
    print(f"Third: {third}")
    assert isinstance(first, str) and len(first) > 0
    assert isinstance(second, str) and len(second) > 0
    assert isinstance(third, str) and len(third) > 0
    assert "key concepts" in second.lower() or "qkv" in second.lower() or "multi-head" in second.lower()
    assert "attention" in third.lower()


def run_ask_structured_test():
    assistant = make_assistant()
    seed_mock_data(assistant)
    question = "What is attention in neural networks?"

    response = assistant.ask_structured(
        question,
        session_id="structured_session",
    )

    print("ask_structured() smoke test passed:")
    print_research_response(question, response)
    assert isinstance(response, ResearchResponse)
    assert isinstance(response.answer, str) and len(response.answer) > 0
    assert response.confidence in {"high", "medium", "low"}
    assert isinstance(response.sources, list) and len(response.sources) > 0
    assert isinstance(response.key_quotes, list)
    assert isinstance(response.follow_up_questions, list) and len(response.follow_up_questions) > 0
    assert "attention" in response.answer.lower()


def run_structured_output_demo_test():
    shutil.rmtree("./research_db", ignore_errors=True)
    assistant = make_assistant()
    seed_mock_data(assistant)

    print(f"\nIndexed: {assistant.get_document_count()} chunks")

    session = "structured_demo"

    print("\n" + "=" * 60)
    print("STEP 1: String response vs Structured response")
    print("=" * 60)

    question = "What is RAG and what are its benefits?"

    print("\n--- String response (ask) ---")
    string_response = assistant.ask(question, "string_test")
    print(f"Type: {type(string_response)}")
    print(f"Response: {string_response[:200]}...")

    print("\n--- Structured response (ask_structured) ---")
    structured_response = assistant.ask_structured(question, "struct_test")
    print(f"Type: {type(structured_response)}")
    print(f"answer:             {structured_response.answer[:100]}...")
    print(f"confidence:         {structured_response.confidence}")
    print(f"sources:            {structured_response.sources}")
    print(f"key_quotes:         {structured_response.key_quotes[:2]}")
    print(f"follow_up_questions: {structured_response.follow_up_questions}")

    print("\n" + "=" * 60)
    print("STEP 2: Use fields in your code")
    print("=" * 60)

    response = assistant.ask_structured("What is the attention mechanism?", session)

    if response.confidence == "high":
        print(f"\n  Confident answer from: {', '.join(response.sources)}")
    else:
        print("\n  Low confidence -- may need more sources")

    print(f"\n  Answer: {response.answer[:200]}")
    print("\n  Suggested follow-ups:")
    for follow_up_question in response.follow_up_questions:
        print(f"    -> {follow_up_question}")

    print("\n" + "=" * 60)
    print("STEP 3: Memory works with structured output too")
    print("=" * 60)

    for question in [
        "What are the components of RAG?",
        "How does the second component work?",
        "Connect everything we discussed to LangChain.",
    ]:
        print(f"\nUser: {question}")
        print_research_response(question, assistant.ask_structured(question, session))

    print("\n" + "=" * 60)
    print("FINAL: What we built across 5 videos")
    print("=" * 60)

    history = assistant._get_session_history(session)
    msg_count = len(history.messages) if hasattr(history, "messages") else len(history)
    print(
        f"""
  Document ingestion    -> {assistant.get_document_count()} chunks indexed
  Sources tracked       -> {assistant.list_sources()}
  Basic retrieval       -> similarity search
  Advanced retrieval    -> multi-query + compression
  Conversation memory   -> {msg_count} messages in session '{session}'
  Structured output     -> ResearchResponse with {len(ResearchResponse.model_fields)} fields

  From raw text to a production-ready research assistant.
  That's the full RAG pipeline.
    """
    )

    shutil.rmtree("./research_db", ignore_errors=True)


if __name__ == "__main__":
    # run_add_documents_test()
    # run_add_text_test()
    # run_build_retriever_test()
    # run_format_docs_for_context_test()
    # run_ask_test()
    # run_ask_failure_test()
    # run_ask_session_test()
    # run_ask_session_scenario_test()
    # run_ask_structured_test()
    run_structured_output_demo_test()

