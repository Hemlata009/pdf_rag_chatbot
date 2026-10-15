import tempfile
import unittest
from pathlib import Path

from src.conversation_memory import ConversationMemory

class ConversationMemoryTests(unittest.TestCase):
    def test_messages_persist_and_recent_turns_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "memory.db")
            memory = ConversationMemory(db, recent_turns=2, summary_trigger_turns=2)
            sid = "session-1"
            for i in range(4):
                memory.add_message(sid, "user", f"question {i}")
                memory.add_message(sid, "assistant", f"answer {i}")
            context = memory.get_context(sid)
            self.assertEqual(len(context["history"]), 2)
            self.assertEqual(context["history"][0]["question"], "question 2")
            self.assertEqual(len(memory.get_full_messages(sid)), 8)
            reopened = ConversationMemory(db, recent_turns=2, summary_trigger_turns=2)
            self.assertEqual(len(reopened.get_full_messages(sid)), 8)

    def test_sessions_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ConversationMemory(str(Path(tmp) / "memory.db"))
            memory.add_message("a", "user", "secret for A")
            memory.add_message("a", "assistant", "answer A")
            self.assertEqual(memory.get_context("b")["history"], [])

    def test_summary_update_uses_callback_and_keeps_recent_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ConversationMemory(str(Path(tmp) / "memory.db"), recent_turns=1, summary_trigger_turns=1)
            for i in range(3):
                memory.add_message("s", "user", f"question {i}")
                memory.add_message("s", "assistant", f"answer {i}")
            memory.maybe_update_summary("s", lambda old, new: (old + " " + "summary of " + new).strip())
            context = memory.get_context("s")
            self.assertTrue(context["summary"])
            self.assertEqual(len(context["history"]), 1)

if __name__ == "__main__":
    unittest.main()
