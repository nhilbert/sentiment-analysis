#!/usr/bin/env python3
"""
Main entry point for German Customer Feedback Analysis
"""

from src.sentiment_analysis import GermanFeedbackAnalyzer

def main():
    """Run the German feedback analysis."""
    analyzer = GermanFeedbackAnalyzer()
    results_df, summary_df = analyzer.run_analysis()
    
    # Display key results
    print(f"\n📊 Analysis completed! {len(results_df)} messages processed.")
    print(f"📁 Results saved to output/ directory")
    
    return results_df, summary_df

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        exit(1)