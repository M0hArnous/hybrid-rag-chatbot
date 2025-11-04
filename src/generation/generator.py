"""
Arabic RAG Generator
Handles Arabic question answering using retrieved context and LLM generation.
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from src.generation.openroute_llm import OpenRouteLLM


class ArabicGenerator:
    """Arabic text generator for RAG-based systems."""

    def __init__(self, model_name: str = "aubmindlab/aragpt2-medium", use_openroute: bool = False):
        """
        Initialize the Arabic text generator.
        Args:
            model_name: Arabic model to use (default = aragpt2)
            use_openroute: Whether to use OpenRoute API instead of local inference
        """
        self.use_openroute = use_openroute

        if use_openroute:
            self.llm = OpenRouteLLM()
        else:
            print(f"[INFO] Loading local model: {model_name}")
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        # Arabic QA prompt template
        self.qa_template = (
            "السؤال:\n{question}\n\n"
            "المعلومات المسترجعة:\n{context}\n\n"
            "بناءً على المعلومات السابقة، أجب بإيجاز ووضوح:\nالإجابة:"
        )

    def generate_response(self, query: str, documents: List[Document]) -> str:
        """
        Generate an Arabic answer using retrieved documents.
        Args:
            query: Arabic user query
            documents: Retrieved context documents
        Returns:
            str: Arabic answer
        """
        # Prepare context
        context = self._prepare_context(documents)

        if not context.strip():
            return "عذرًا، لم أجد معلومات كافية للإجابة على هذا السؤال."

        # Build Arabic prompt
        prompt = self.qa_template.format(question=query.strip(), context=context.strip())

        # ✅ Truncate prompt to fit model's max length (to avoid IndexError)
        max_len = getattr(self.model.config, "n_positions", 1024)
        input_ids = self.tokenizer.encode(prompt, truncation=True, max_length=max_len - 256)
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        # ✅ Use OpenRoute API if enabled
        if self.use_openroute:
            print("[INFO] Generating using OpenRoute API...")
            return self.llm(prompt)

        print("[INFO] Generating locally...")

        # ✅ Safe generation config
        response = self.generator(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Extract and clean response
        generated_text = response[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()
        answer = answer.replace("\n\n", "\n").strip()

        if not answer:
            answer = "عذرًا، لم أتمكن من توليد إجابة مناسبة."

        return answer

    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Combine context from retrieved documents into one Arabic text block.
        Truncate if too long to prevent overflow.
        """
        context_parts = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"مستند {i+1}")
            snippet = doc.page_content.strip()
            if snippet:
                context_parts.append(f"[{i+1}] المصدر: {source}\n{snippet}\n")

        combined = "\n".join(context_parts)

        # ✅ Limit context to 2000 characters for stability
        if len(combined) > 2000:
            combined = combined[:2000] + "...\n(تم اختصار المحتوى الطويل للحفاظ على الدقة.)"

        return combined


class ArabicRAG:
    """Full Arabic Retrieval-Augmented Generation pipeline."""

    def __init__(self, retriever, generator):
        """
        Initialize the Arabic RAG system.
        Args:
            retriever: A retriever object that supports get_relevant_documents()
            generator: An ArabicGenerator instance
        """
        self.retriever = retriever
        self.generator = generator

    def query(self, query: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        Args:
            query: Arabic user query
        Returns:
            dict: { query, response, documents }
        """
        print(f"[INFO] Retrieving context for: {query}")
        documents = self.retriever.get_relevant_documents(query)

        print(f"[INFO] Retrieved {len(documents)} documents")
        response = self.generator.generate_response(query, documents)

        return {
            "query": query,
            "response": response,
            "documents": documents
        }
