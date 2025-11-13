from __future__ import annotations

import logging
from threading import Lock
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

import torch
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

from app.core.config import AppSettings, get_settings

logger = logging.getLogger(__name__)


class RAGService:
    """Manage ingestion and querying over legal PDFs using LlamaIndex and Qwen."""

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._callback_manager = callback_manager
        self._llm = None
        self._embed_model = None
        self._index: Optional[VectorStoreIndex] = None
        self._lock = Lock()

    def ingest(self) -> Dict[str, int]:
        """Load PDFs, rebuild the vector index, and persist it to disk."""
        with self._lock:
            documents = SimpleDirectoryReader(
                input_dir=str(self.settings.resolved_data_dir),
                recursive=True,
            ).load_data()

            if not documents:
                msg = (
                    f"No documents found in {self.settings.resolved_data_dir}. "
                    "Place the POCSO and BNSS PDFs in this directory before ingesting."
                )
                logger.warning(msg)
                raise FileNotFoundError(msg)

            with self._settings_context():
                index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True,
                )
                index.storage_context.persist(persist_dir=str(self.settings.resolved_persist_dir))

            self._index = index
            logger.info("Persisted vector index with %s documents", len(documents))
            return {"documents_indexed": len(documents)}

    def query(self, question: str, similarity_top_k: int = 3) -> Dict[str, object]:
        """Answer a question using the indexed documents and Qwen LLM."""
        if not question.strip():
            raise ValueError("Question must be a non-empty string.")

        with self._settings_context():
            index = self._ensure_index_loaded()
            query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
            response = query_engine.query(question)

        sources: List[Dict[str, str]] = []
        for node in response.source_nodes:
            metadata = node.metadata or {}
            sources.append(
                {
                    "id": node.node_id,
                    "score": node.score or 0.0,
                    "file_name": metadata.get("file_name") or metadata.get("filename") or "",
                    "page_number": metadata.get("page_label") or metadata.get("page_number") or "",
                    "text": node.get_content(),
                }
            )

        return {
            "answer": str(response),
            "sources": sources,
        }

    def _ensure_index_loaded(self) -> VectorStoreIndex:
        with self._lock:
            if self._index is not None:
                return self._index

            persist_dir = self.settings.resolved_persist_dir
            if not any(persist_dir.glob("**/*")):
                logger.info("No persisted index found; triggering ingestion.")
                self.ingest()
                return self._index  # type: ignore[return-value]

            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            with self._settings_context():
                self._index = load_index_from_storage(
                    storage_context=storage_context,
                )
            logger.info("Loaded persisted index from %s", persist_dir)
            return self._index

    @contextmanager
    def _settings_context(self) -> Generator[None, None, None]:
        llm = self._ensure_llm()
        embed_model = self._ensure_embedding_model()
        previous_llm = Settings.llm
        previous_embed = Settings.embed_model
        previous_callback = Settings.callback_manager
        Settings.llm = llm
        Settings.embed_model = embed_model
        if self._callback_manager is not None:
            Settings.callback_manager = self._callback_manager
        try:
            yield
        finally:
            Settings.llm = previous_llm
            Settings.embed_model = previous_embed
            Settings.callback_manager = previous_callback

    def _ensure_llm(self) -> HuggingFaceLLM:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _ensure_embedding_model(self) -> HuggingFaceEmbedding:
        if self._embed_model is None:
            self._embed_model = self._create_embedding_model()
        return self._embed_model

    def _create_llm(self) -> HuggingFaceLLM:
        model_kwargs: Dict[str, object] = {"trust_remote_code": True}
        quantization_config: Optional[BitsAndBytesConfig] = None
        device_map = self.settings.device_map
        if self.settings.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["torch_dtype"] = torch.bfloat16
            device_map = device_map or "auto"
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                max_memory = {}
                for i in range(num_gpus):
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    reserved = int(total_mem * 0.1)
                    max_memory[f"cuda:{i}"] = f"{(total_mem - reserved) // (1024**2)}MiB"
                model_kwargs["max_memory"] = max_memory
            else:
                device_map = "cpu"
        else:
            model_kwargs["torch_dtype"] = torch.float16
            if torch.cuda.is_available():
                device_map = device_map or "auto"
            else:
                device_map = device_map or "cpu"

        # Use a concise system prompt to ground the assistant in legal context.
        system_prompt = (
            "You are a legal assistant specializing in Indian legislation. "
            "Use the provided context to answer questions about the POCSO and BNSS Acts. "
            "If the answer is not in the context, say you do not know."
        )

        return HuggingFaceLLM(
            model_name=self.settings.model_name,
            tokenizer_name=self.settings.model_name,
            generate_kwargs={
                "temperature": self.settings.temperature,
                "max_new_tokens": self.settings.max_new_tokens,
            },
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left", "truncation_side": "left"},
            system_prompt=system_prompt,
            device_map=device_map,
            messages_to_prompt=lambda messages: self._messages_to_prompt(messages, system_prompt),
            completion_to_prompt=lambda completion: completion,
        )

    def _create_embedding_model(self) -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(model_name=self.settings.embedding_model_name)

    @staticmethod
    def _messages_to_prompt(messages: List[ChatMessage], system_prompt: str) -> str:
        prompt_parts: List[str] = [system_prompt]
        for message in messages:
            role = "User" if message.role == MessageRole.USER else "Assistant"
            prompt_parts.append(f"{role}: {message.content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

