
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# Data Models
# ============================================================
class ResearchResponse(BaseModel):
    """Structured response from the research assistant."""

    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium, or low based on source quality")
    sources: List[str] = Field(description="List of source documents used")
    key_quotes: List[str] = Field(
        description="Relevant quotes from sources", default=[]
    )
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")

# ============================================================
# Research Assistant Class
# ============================================================

class AIResearchAssistant:
    """AI Research Assistant with document ingestion and retrieval."""

    def __init__(
        self,
        persist_directory: str = "./research_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_directory = persist_directory

        # 1. Embeddings - turns text into vectors
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 2. Splitter - breaks big docs into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # 3. Vector store - stores and searches embeddings
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_docs",
        )

        self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

        print(f"Research Assistant initialized")
        print(f"  Vector store: {persist_directory}")
        print(f"  Documents indexed: {self.vectorstore._collection.count()}")

    def add_documents(
        self,
        documents: List[Document],
        source_name: Optional[str] = None,
    ) -> int:
        """Add documents to the research database."""

        # Tag with source name
        if source_name:
            for doc in documents:
                doc.metadata["source"] = source_name

        # Split into chunks
        chunks = self.splitter.split_documents(documents)

        # Timestamp each chunk
        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()

        # Store in vector DB
        self.vectorstore.add_documents(chunks)

        print(f"Added {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)

    def add_text(self, text: str, source: str, metadata: dict = None) -> int:
        """Add a single text string as a document."""
        doc = Document(
            page_content=text, metadata={"source": source, **(metadata or {})}
        )
        return self.add_documents([doc])

    def add_texts(self, texts: List[str], source: str) -> int:
        """Add multiple text strings from the same source."""
        docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
        return self.add_documents(docs)

    def get_document_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.vectorstore._collection.count()

    def list_sources(self) -> List[str]:
        """List all unique sources in the database."""
        results = self.vectorstore._collection.get()
        sources = set()
        for metadata in results.get("metadatas", []):
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
        return sorted(list(sources))

if __name__ == "__main__":
    assistant = AIResearchAssistant(persist_directory="./research_db")
    print(f"Total indexed chunks: {assistant.get_document_count()}")
    print(f"Sources: {assistant.list_sources()}")