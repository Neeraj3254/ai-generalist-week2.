"""
Test Suite for ETL Pipeline
Week 2, Day 1 - AI Generalist Training
"""

import unittest
from datetime import datetime
from etl_pipeline import (
    DataExtractor,
    DataTransformer,
    DataLoader,
    ETLPipeline,
    Document,
    TransformedRecord
)


class TestDataExtractor(unittest.TestCase):
    """Test the data extraction layer"""
    
    def setUp(self):
        self.extractor = DataExtractor()
    
    def test_scrape_valid_url(self):
        """Test scraping a valid URL"""
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        doc = self.extractor.scrape_url(url)
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.url, url)
        self.assertIsInstance(doc.content, str)
        self.assertGreater(len(doc.content), 100)
        self.assertIn('word_count', doc.metadata)
    
    def test_scrape_invalid_url(self):
        """Test handling of invalid URL"""
        url = "https://this-url-definitely-does-not-exist-12345.com"
        doc = self.extractor.scrape_url(url)
        
        self.assertIsNone(doc)
    
    def test_extract_batch(self):
        """Test batch extraction"""
        urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://this-does-not-exist.com",
            "https://en.wikipedia.org/wiki/Machine_learning"
        ]
        
        docs = self.extractor.extract_batch(urls)
        
        # Should get 2 valid docs (one URL is invalid)
        self.assertEqual(len(docs), 2)


class TestDataTransformer(unittest.TestCase):
    """Test the data transformation layer"""
    
    def setUp(self):
        self.transformer = DataTransformer(chunk_size=500, chunk_overlap=50)
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk_size"""
        text = "This is a short text."
        chunks = self.transformer.chunk_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_chunk_long_text(self):
        """Test chunking long text"""
        text = "This is a sentence. " * 100  # ~2000 chars
        chunks = self.transformer.chunk_text(text)
        
        self.assertGreater(len(chunks), 1)
        # Check overlap exists
        for i in range(len(chunks) - 1):
            self.assertTrue(chunks[i][-20:] in chunks[i + 1] or 
                          chunks[i + 1][:20] in chunks[i])
    
    def test_transform_document(self):
        """Test document transformation"""
        doc = Document(
            url="https://example.com",
            title="Test Document",
            content="This is test content. " * 100,
            metadata={'source_type': 'test'},
            timestamp=datetime.now()
        )
        
        records = self.transformer.transform_document(doc)
        
        self.assertGreater(len(records), 0)
        self.assertIsInstance(records[0], TransformedRecord)
        self.assertEqual(records[0].metadata['source_url'], doc.url)
        self.assertEqual(records[0].metadata['chunk_index'], 0)


class TestDataLoader(unittest.TestCase):
    """Test the data loading layer"""
    
    def setUp(self):
        self.loader = DataLoader(collection_name="test_collection")
    
    def test_generate_embedding(self):
        """Test embedding generation"""
        text = "This is a test sentence for embedding."
        embedding = self.loader.generate_embedding(text)
        
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)
    
    def test_load_and_search(self):
        """Test loading records and searching"""
        # Create test records
        records = [
            TransformedRecord(
                id="test_1",
                text="Machine learning is a subset of artificial intelligence.",
                metadata={'source': 'test'}
            ),
            TransformedRecord(
                id="test_2",
                text="Deep learning uses neural networks with multiple layers.",
                metadata={'source': 'test'}
            )
        ]
        
        # Load records
        loaded = self.loader.load_records(records)
        self.assertEqual(loaded, 2)
        
        # Search
        results = self.loader.search("What is machine learning?", top_k=1)
        self.assertGreater(len(results), 0)
        self.assertIn('machine learning', results[0]['text'].lower())


class TestETLPipeline(unittest.TestCase):
    """Test the complete ETL pipeline"""
    
    def test_full_pipeline(self):
        """Test running the complete pipeline"""
        urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        ]
        
        pipeline = ETLPipeline()
        stats = pipeline.run(urls)
        
        self.assertEqual(stats['urls_total'], 1)
        self.assertGreater(stats['documents_extracted'], 0)
        self.assertGreater(stats['records_created'], 0)
        self.assertGreater(stats['records_loaded'], 0)
        self.assertIsNotNone(stats['start_time'])
        self.assertIsNotNone(stats['end_time'])


def run_tests():
    """Run all tests and display results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataTransformer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestETLPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()