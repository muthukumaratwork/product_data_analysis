#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced ProductAnalyzer with configurable products per request.

This demonstrates the difference between:
1. Traditional approach: 1 product per API request
2. Batch approach: Multiple products per API request (e.g., 5 products per request)

Example usage scenarios:
- 5 products per request with 50 batch size
- 10 products per request with 100 batch size
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to Python path to import product_analyzer
sys.path.append(str(Path(__file__).parent))

from product_analyzer import ProductAnalyzer


def demo_traditional_approach():
    """Demo with 1 product per API request (traditional approach)"""
    print("=== Traditional Approach: 1 Product per API Request ===")
    
    analyzer = ProductAnalyzer(
        api_key=os.getenv('OPENAI_API_KEY'),
        max_workers=5,
        products_per_request=1  # Traditional approach
    )
    
    start_time = time.time()
    
    # Process first 10 records with traditional approach
    analyzer.process_csv(
        input_file="test_batch_input.csv",
        output_file="output_traditional.csv",
        batch_size=10,
        start_record=1,
        end_record=10,
        delay=0.1
    )
    
    end_time = time.time()
    print(f"Traditional approach completed in {end_time - start_time:.2f} seconds")
    print()


def demo_batch_approach():
    """Demo with 5 products per API request (batch approach)"""
    print("=== Batch Approach: 5 Products per API Request ===")
    
    analyzer = ProductAnalyzer(
        api_key=os.getenv('OPENAI_API_KEY'),
        max_workers=5,
        products_per_request=5  # Batch approach
    )
    
    start_time = time.time()
    
    # Process first 10 records with batch approach
    analyzer.process_csv(
        input_file="test_batch_input.csv",
        output_file="output_batch.csv",
        batch_size=10,
        start_record=1,
        end_record=10,
        delay=0.1
    )
    
    end_time = time.time()
    print(f"Batch approach completed in {end_time - start_time:.2f} seconds")
    print()


def demo_large_batch_scenario():
    """Demo the specific scenario requested: 5 products per request, 50 batch size"""
    print("=== Requested Scenario: 5 Products per Request, 50 Batch Size ===")
    
    analyzer = ProductAnalyzer(
        api_key=os.getenv('OPENAI_API_KEY'),
        max_workers=10,
        products_per_request=5  # 5 products per API request
    )
    
    start_time = time.time()
    
    # Process all records with the requested configuration
    analyzer.process_csv(
        input_file="test_batch_input.csv",
        output_file="output_scenario.csv",
        batch_size=50,  # 50 batch size
        delay=0.05
    )
    
    end_time = time.time()
    print(f"Requested scenario completed in {end_time - start_time:.2f} seconds")
    print()


def main():
    """Run all demos"""
    print("Product Analyzer Batch Processing Demo")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: Please set the OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Check if test input file exists
    if not Path("test_batch_input.csv").exists():
        print("ERROR: test_batch_input.csv not found")
        sys.exit(1)
    
    try:
        # Run demonstrations
        demo_traditional_approach()
        demo_batch_approach()
        demo_large_batch_scenario()
        
        print("Demo completed! Check the output files:")
        print("- output_traditional.csv (1 product per request)")
        print("- output_batch.csv (5 products per request)")
        print("- output_scenario.csv (5 products per request, 50 batch size)")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
