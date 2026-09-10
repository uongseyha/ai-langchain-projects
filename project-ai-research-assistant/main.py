
import shutil

from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from data import seed_mock_data

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

    def _build_retriever(self, use_advanced: bool = False):
        """Build retriever -- basic or advanced"""

        # Base: simple similarity search
        base_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        if not use_advanced:
            return base_retriever

        # Multi-query: LLM generates multiple search queries
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )

        return multi_retriever

    def _format_docs_for_context(self, docs) -> str:
        """Format retrieved documents into a string for the prompt."""
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.session_store:
            self.session_store[session_id].clear()
            print(f"Cleared session: {session_id}")

    def get_session_messages(self, session_id: str) -> list:
        """Get conversation history as readable dicts."""
        if session_id not in self.session_store:
            return []
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in self.session_store[session_id].messages
        ]
    
    def ask(
        self, question: str, session_id: str = "default", use_advanced: bool = True
    ) -> str:
        """Ask a question against the research documents."""

        history = self._get_session_history(session_id)

        # Step 1: Use basic or advanced retriever
        retriever = self._build_retriever(use_advanced=use_advanced)
        docs = retriever.invoke(question)

        # Step 2: Format retrieved docs for context
        context = self._format_docs_for_context(docs)

        # Step 3: Build the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI Research Assistant. Answer questions
    based ONLY on the provided context documents.

    Rules:
    1. Only use information from the context below
    2. If the context doesn't have the answer, say so
    3. Cite which sources you used (e.g. "According to Source 1...")
    4. Rate your confidence: high, medium, or low""",
                ),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    """Context documents:

    {context}

    Question: {question}

    Provide a clear answer with source citations.""",
                ),
            ]
        )

        # Step 4: Build and run the chain
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "history": history.messages[-10:],  # last 10 messages for context
            }
        )

        # save this Q&A to history
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response))

        return response

if __name__ == "__main__":
    shutil.rmtree("./research_db", ignore_errors=True)
    assistant = AIResearchAssistant(persist_directory="./research_db")
    seed_mock_data(assistant)



    # Cleanup
    shutil.rmtree("./research_db", ignore_errors=True)