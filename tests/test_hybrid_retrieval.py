import unittest

from src.retrieval.hybrid_retrieval import bm25_retrieve, hybrid_retrieve, reciprocal_rank_fusion


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": "a", "text": "The intern built a Power BI dashboard for revenue", "metadata": {"source": "a.pdf", "page": 1}},
            {"chunk_id": "b", "text": "SQL was used to clean transactional data and create reports", "metadata": {"source": "b.pdf", "page": 2}},
            {"chunk_id": "c", "text": "Revenue dashboard metrics include conversion and retention", "metadata": {"source": "c.pdf", "page": 3}},
        ]

    def test_bm25_ranks_exact_lexical_match(self):
        results = bm25_retrieve("Power BI dashboard", self.chunks, top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], "a")
        self.assertIn("bm25_score", results[0])

    def test_bm25_handles_empty_queries(self):
        self.assertEqual(bm25_retrieve("   ", self.chunks), [])
        self.assertEqual(bm25_retrieve("something", [], top_k=4), [])

    def test_rrf_deduplicates_and_combines_rankings(self):
        dense = [self.chunks[0], self.chunks[1]]
        lexical = [self.chunks[1], self.chunks[2]]
        fused = reciprocal_rank_fusion([dense, lexical], top_k=3, rrf_k=60)
        ids = [item["chunk_id"] for item in fused]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "b")
        self.assertIn("fusion_score", fused[0])

    def test_hybrid_retrieval_retains_source_metadata(self):
        dense = [{**self.chunks[1], "similarity": 0.8}]
        fused = hybrid_retrieve("SQL transactional data", dense, self.chunks, top_k=3)
        self.assertTrue(fused)
        self.assertTrue(all("metadata" in item and "chunk_id" in item for item in fused))

    def test_fusion_rejects_invalid_rrf_constant(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], rrf_k=0)


if __name__ == "__main__":
    unittest.main()
