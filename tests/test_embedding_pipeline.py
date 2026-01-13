import unittest
import pandas as pd
from src.embedding_pipeline import stratified_sample, chunk_texts, ensure_complaint_id


class TestEmbeddingPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "product": ["Credit card", "Savings account", "Credit card", "Money transfers"],
            "clean_narrative": [
                "This is a test complaint about credit card issues.",
                "My savings account was closed without notice.",
                "Another credit card complaint with more details.",
                "Money transfer failed and I lost money."
            ]
        })

    def test_ensure_complaint_id(self):
        df_id = ensure_complaint_id(self.df)
        self.assertIn("complaint_id", df_id.columns)
        self.assertEqual(len(df_id["complaint_id"].unique()), len(df_id))

    def test_stratified_sample(self):
        sample = stratified_sample(self.df, "product", 4, ["Credit card", "Savings account", "Money transfers"])
        self.assertEqual(len(sample), 4)
        self.assertTrue(set(sample["product"]).issubset(set(self.df["product"])))

    def test_chunk_texts(self):
        df_id = ensure_complaint_id(self.df)
        chunks = chunk_texts(df_id, chunk_size=50, chunk_overlap=10)
        self.assertGreater(len(chunks), 0)
        self.assertIn("chunk", chunks[0])
        self.assertIn("product", chunks[0])
        self.assertIn("chunk_index", chunks[0])
        self.assertIn("complaint_id", chunks[0])


if __name__ == "__main__":
    unittest.main()