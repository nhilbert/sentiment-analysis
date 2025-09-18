#!/usr/bin/env python3
"""
Test runner script for the sentiment analysis project
"""

import sys
import os
import unittest

def run_all_tests():
    """Discover and run all tests"""
    # Add src to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print(f"{'='*50}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)