#!/usr/bin/env python3
"""
Basic tests for German Feedback Analyzer
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analysis import GermanFeedbackAnalyzer


class TestGermanFeedbackAnalyzer(unittest.TestCase):
    """Test cases for GermanFeedbackAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = GermanFeedbackAnalyzer()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'feedback': [
                'Der Service war ausgezeichnet!',
                'Die App ist schlecht.',
                'Funktioniert gut, danke.'
            ]
        })
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsInstance(self.analyzer, GermanFeedbackAnalyzer)
        self.assertIsInstance(self.analyzer.capabilities, dict)
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        # Rename columns to match expected format
        test_df = self.test_data.copy()
        test_df.columns = ['date', 'message']
        
        cleaned_df = self.analyzer.clean_data(test_df)
        
        self.assertEqual(len(cleaned_df), 3)
        self.assertIn('nachricht', cleaned_df.columns)
        self.assertIn('year_month', cleaned_df.columns)
    
    def test_german_sentiment_analysis(self):
        """Test German sentiment analysis"""
        texts = [
            'Der Service war ausgezeichnet!',
            'Die App ist schlecht.',
            'Neutral text here.'
        ]
        
        scores, categories = self.analyzer._german_sentiment_analysis(texts)
        
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(categories), 3)
        self.assertIn(categories[0], ['positive', 'negative', 'neutral'])
    
    def test_create_embeddings(self):
        """Test text embedding creation"""
        texts = ['Test text', 'Another test']
        embeddings = self.analyzer.create_embeddings(texts)
        
        self.assertEqual(embeddings.shape[0], 2)
        self.assertGreater(embeddings.shape[1], 0)
    
    def test_german_stopwords(self):
        """Test German stopwords functionality"""
        stopwords = self.analyzer._get_german_stopwords()
        
        self.assertIsInstance(stopwords, set)
        self.assertIn('der', stopwords)
        self.assertIn('und', stopwords)
    
    def test_cluster_labeling(self):
        """Test cluster label generation"""
        texts = ['Support problem', 'Login issue', 'Payment error']
        labels = np.array([0, 0, 1])
        
        cluster_meta = self.analyzer.generate_cluster_labels(texts, labels)
        
        self.assertIsInstance(cluster_meta, dict)
        self.assertIn(0, cluster_meta)
        self.assertIn('label', cluster_meta[0])


class TestDataValidation(unittest.TestCase):
    """Test data validation and error handling"""
    
    def test_missing_file(self):
        """Test handling of missing input file"""
        analyzer = GermanFeedbackAnalyzer('nonexistent.xlsx')
        
        with self.assertRaises(FileNotFoundError):
            analyzer.load_data()
    
    def test_empty_dataframe(self):
        """Test handling of empty data"""
        analyzer = GermanFeedbackAnalyzer()
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        try:
            result = analyzer.clean_data(empty_df)
            self.assertEqual(len(result), 0)
        except Exception as e:
            # Expected to fail, but shouldn't crash
            self.assertIsInstance(e, (ValueError, KeyError))


class TestSentimentMethods(unittest.TestCase):
    """Test different sentiment analysis methods"""
    
    def setUp(self):
        self.analyzer = GermanFeedbackAnalyzer()
        self.german_texts = [
            'Das ist ausgezeichnet!',
            'Sehr schlecht und frustrierend.',
            'Okay, funktioniert normal.'
        ]
    
    def test_enhanced_vader_sentiment(self):
        """Test enhanced VADER with German mappings"""
        if self.analyzer.capabilities.get('vader'):
            scores, categories = self.analyzer._enhanced_vader_sentiment_analysis(self.german_texts)
            
            self.assertEqual(len(scores), 3)
            self.assertEqual(len(categories), 3)
            # First text should be positive due to 'ausgezeichnet'
            self.assertGreater(scores[0], 0)
    
    def test_sentiment_score_range(self):
        """Test sentiment scores are in valid range"""
        scores, categories = self.analyzer._german_sentiment_analysis(self.german_texts)
        
        for score in scores:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)
        
        for category in categories:
            self.assertIn(category, ['positive', 'negative', 'neutral'])


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGermanFeedbackAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentMethods))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)