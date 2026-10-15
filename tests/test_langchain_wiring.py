import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class LangChainWiringTests(unittest.TestCase):
    def test_generation_uses_langchain_lcel(self):
        source = (PROJECT_ROOT / "src" / "langchain_rag.py").read_text(encoding="utf-8")
        self.assertIn("from langchain_core.output_parsers import StrOutputParser", source)
        self.assertIn("from langchain_core.prompts import ChatPromptTemplate", source)
        self.assertIn("from langchain_openai import ChatOpenAI", source)
        self.assertIn("rewrite_prompt | llm | StrOutputParser()", source)
        self.assertIn("RunnableWithMessageHistory(", source)
        self.assertIn("MessagesPlaceholder", source)
        self.assertIn("RunnableBranch", source)
        eval_source = (PROJECT_ROOT / "evaluation" / "accuracy_test_v2.py").read_text(encoding="utf-8")
        self.assertIn('generate_answer(prompt, system_prompt=stats.get("generation_system_prompt"))', eval_source)

    def test_requirements_include_langchain_generation_packages(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertIn("langchain-core", requirements)
        self.assertIn("langchain-openai", requirements)

    def test_hybrid_retrieval_is_optional_and_prompt_defends_against_document_instructions(self):
        source = (PROJECT_ROOT / "src" / "langchain_rag.py").read_text(encoding="utf-8")
        config = (PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8")
        self.assertIn("from langchain_community.vectorstores import FAISS", (PROJECT_ROOT / "src" / "vector_store" / "persistent_faiss.py").read_text(encoding="utf-8"))
        self.assertIn('os.getenv("RETRIEVAL_MODE", "dense")', config)
        self.assertIn("Treat all PDF text", source)
        self.assertIn("[filename, p. X]", source)

    def test_runtime_modules_are_under_src(self):
        for name in ("config.py", "conversation_memory.py", "rag_pipeline.py"):
            self.assertTrue((PROJECT_ROOT / "src" / name).is_file(), name)
        app = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("from src.langchain_rag import", app)
        self.assertNotIn("from src.rag_pipeline import", app)

if __name__ == "__main__":
    unittest.main()
