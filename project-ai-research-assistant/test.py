from langchain_core.documents import Document
from main import AIResearchAssistant


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


if __name__ == "__main__":
    run_add_documents_test()
    run_add_text_test()

