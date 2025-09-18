#!/usr/bin/env python3
"""
Main entry point for German Customer Feedback Analysis
"""

from src.sentiment_analysis import GermanFeedbackAnalyzer

def main():
    """Run the German feedback analysis."""
    analyzer = GermanFeedbackAnalyzer()
    results_df, summary_df = analyzer.run_analysis()
    return results_df, summary_df

if __name__ == "__main__":
    main()