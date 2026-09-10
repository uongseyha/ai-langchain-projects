from langchain_core.documents import Document
from main import AIResearchAssistant
from data import mock_research_data, seed_mock_data


def make_assistant():
    return AIResearchAssistant(persist_directory="./research_db")


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


if __name__ == "__main__":
    # run_add_documents_test()
    # run_add_text_test()
    # run_build_retriever_test()
    # run_format_docs_for_context_test()
    # run_ask_test()
    # run_ask_failure_test()
    # run_ask_session_test()
    run_ask_session_scenario_test()

