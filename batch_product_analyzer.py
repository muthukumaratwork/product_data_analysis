#!/usr/bin/env python3
"""
-------------------------------------------------------
Batch Product Analyzer Script - Separate Implementation

How to run:
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Basic batch processing (5 products per request)
python batch_product_analyzer.py input.csv output.csv --products-per-request 5

# Example scenario: 5 products per request, 50 batch size
python batch_product_analyzer.py input.csv output.csv --products-per-request 5 --batch-size 50 --threads 10

# Large dataset processing
python batch_product_analyzer.py large_file.csv output.csv --products-per-request 8 --batch-size 200 --threads 20

Model configured is gpt-4
-------------------------------------------------------

Batch Product Analyzer Script

This script is specifically designed for batch processing multiple products 
in single OpenAI API requests for maximum efficiency.

Features:
- Multiple products per API request (configurable)
- Optimized for large datasets
- Multithreading for concurrent batch processing
- Progress tracking and logging
- Resume capability
- Cost-effective API usage

Required environment variable:
- OPENAI_API_KEY: Your OpenAI API key

Usage:
    python batch_product_analyzer.py input.csv output.csv [options]
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import openai
from openai import OpenAI


class BatchProductAnalyzer:
    """Specialized product analyzer for batch processing multiple products per API request."""
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 10, products_per_request: int = 5):
        """Initialize the BatchProductAnalyzer."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.products_per_request = max(1, min(products_per_request, 10))  # Clamp between 1-10
        self.batch_prompt_template = self._load_batch_prompt_template()
        self.max_workers = max_workers
        self.write_lock = Lock()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('batch_product_analyzer.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_batch_prompt_template(self) -> str:
        """Load the batch prompt template from the text file."""
        script_dir = Path(__file__).parent
        prompt_file = script_dir / "batch_product_analysis_prompt.txt"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Create the prompt file if it doesn't exist
            self._create_batch_prompt_template(prompt_file)
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _create_batch_prompt_template(self, prompt_file: Path) -> None:
        """Create the batch prompt template file."""
        template_content = """You are a product analysis expert. Given multiple products with their names and UPC numbers, analyze each product and extract the following information in a structured format.

Products to analyze:
{products_list}

Please analyze each product and provide the following details in JSON format as an array of objects:

[
  {{
    "UPC": "The UPC number (same as input)",
    "product_name": "The standardized product name",
    "brand": "The brand name of the product",
    "product_type": "The category/type of product (e.g., electronics, food, clothing, etc.)",
    "color_desc": "Color description if applicable, or 'N/A' if not relevant",
    "dimension": "Physical dimensions if known, or 'N/A' if not available",
    "pc_count": "Number of pieces/units in package, or '1' if single item",
    "capacity": "Storage capacity, volume, or size specification, or 'N/A' if not applicable",
    "configuration": "Product configuration details, or 'N/A' if not applicable",
    "key_feature": "Main distinguishing features of the product"
  }},
  // ... repeat for each product
]

Important instructions:
- Analyze each product in the order provided
- If specific information is not available or cannot be determined from the product name and UPC, use "N/A"
- For pc_count, use "1" if it's a single item and no count is specified
- Be as accurate as possible based on the product name provided
- Return only the JSON array, no additional text
- Ensure all fields are filled with either appropriate values or "N/A"
- The response must contain exactly {num_products} product objects in the same order as the input"""
        
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
    
    def analyze_product_batch(self, products: List[Tuple[int, Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Analyze multiple products in a single OpenAI API call.
        
        Args:
            products: List of tuples (record_index, product_dict)
            
        Returns:
            List of analyzed product results
        """
        try:
            # Prepare products list for the prompt
            products_list = ""
            for i, (record_index, product) in enumerate(products, 1):
                products_list += f"{i}. Product Name: {product['product_name']}\n   UPC Number: {product['upc_number']}\n\n"
            
            # Format the batch prompt
            prompt = self.batch_prompt_template.format(
                products_list=products_list.strip(),
                num_products=len(products)
            )
            
            # Calculate max tokens based on number of products
            max_tokens = min(4000, 350 * len(products) + 300)
            
            # Call OpenAI API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful product analysis assistant specialized in batch processing."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=0.1
                    )
                    break
                except Exception as api_error:
                    if attempt == max_retries - 1:
                        raise api_error
                    record_indices = [r for r, _ in products]
                    self.logger.warning(f"API call failed for batch {record_indices}, attempt {attempt + 1}: {api_error}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Extract the response content
            response_content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                results = json.loads(response_content)
                if not isinstance(results, list):
                    raise ValueError("Expected a list of results")
                
                # Add original input data and record indices
                final_results = []
                for i, (record_index, product) in enumerate(products):
                    if i < len(results):
                        result = results[i]
                        result['original_product_name'] = product['product_name']
                        result['original_upc_number'] = product['upc_number']
                        result['record_index'] = record_index
                        final_results.append(result)
                    else:
                        # Fallback if not enough results returned
                        self.logger.warning(f"Missing result for record {record_index}, using fallback")
                        final_results.append(self._create_fallback_result(
                            product['product_name'], product['upc_number'], record_index))
                
                return final_results
                
            except (json.JSONDecodeError, ValueError) as e:
                record_indices = [r for r, _ in products]
                self.logger.warning(f"Could not parse JSON response for batch {record_indices}: {e}")
                # Return fallback results for all products
                return [self._create_fallback_result(product['product_name'], product['upc_number'], record_index) 
                       for record_index, product in products]
                
        except Exception as e:
            record_indices = [r for r, _ in products]
            self.logger.error(f"Error analyzing product batch {record_indices}: {str(e)}")
            # Return fallback results for all products
            return [self._create_fallback_result(product['product_name'], product['upc_number'], record_index) 
                   for record_index, product in products]
    
    def _create_fallback_result(self, product_name: str, upc_number: str, record_index: int = 0) -> Dict[str, str]:
        """Create a fallback result when API call fails."""
        return {
            'original_product_name': product_name,
            'original_upc_number': upc_number,
            'UPC': upc_number,
            'product_name': product_name,
            'brand': 'N/A',
            'product_type': 'N/A',
            'color_desc': 'N/A',
            'dimension': 'N/A',
            'pc_count': '1',
            'capacity': 'N/A',
            'configuration': 'N/A',
            'key_feature': 'N/A',
            'record_index': record_index
        }
    
    def process_batch_group(self, batch_group: List[Tuple[int, Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Process a group of products by splitting them into API-sized batches.
        
        Args:
            batch_group: List of tuples (record_index, product_dict)
            
        Returns:
            List of analyzed product results
        """
        results = []
        
        # Split batch_group into sub-batches based on products_per_request
        api_batches = []
        for i in range(0, len(batch_group), self.products_per_request):
            api_batch = batch_group[i:i + self.products_per_request]
            api_batches.append(api_batch)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit API batch tasks
            future_to_batch = {
                executor.submit(self.analyze_product_batch, api_batch): api_batch
                for api_batch in api_batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                api_batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    record_indices = [r for r, _ in api_batch]
                    self.logger.error(f"Batch processing failed for records {record_indices}: {e}")
                    # Create fallback results for all products in batch
                    for record_index, product in api_batch:
                        fallback = self._create_fallback_result(
                            product['product_name'], 
                            product['upc_number'], 
                            record_index
                        )
                        results.append(fallback)
        
        # Sort results by record_index to maintain order
        results.sort(key=lambda x: x['record_index'])
        return results
    
    def process_csv(self, input_file: str, output_file: str, batch_size: int = 50, 
                   start_record: Optional[int] = None, end_record: Optional[int] = None,
                   delay: float = 0.05) -> None:
        """
        Process a CSV file with batch processing optimized for multiple products per API request.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            batch_size: Number of records to process in each group
            start_record: Start processing from this record number (1-based)
            end_record: Stop processing at this record number (inclusive, 1-based)
            delay: Delay between batch groups to avoid rate limiting
        """
        try:
            # Read input CSV with record range
            products = self._read_csv_range(input_file, start_record, end_record)
            
            if not products:
                raise ValueError("No products found in specified range")
            
            # Validate required columns
            required_columns = ['product_name', 'upc_number']
            if not all(col in products[0][1].keys() for col in required_columns):
                raise ValueError(f"Input CSV must contain columns: {required_columns}")
            
            total_products = len(products)
            start_idx = products[0][0] if products else 0
            end_idx = products[-1][0] if products else 0
            
            self.logger.info(f"Processing {total_products} products (records {start_idx} to {end_idx})")
            self.logger.info(f"Using {self.products_per_request} products per API request")
            self.logger.info(f"Using {self.max_workers} threads with batch size {batch_size}")
            
            # Calculate efficiency metrics
            total_api_calls = (total_products + self.products_per_request - 1) // self.products_per_request
            self.logger.info(f"Estimated API calls: {total_api_calls} (vs {total_products} with single-product approach)")
            
            # Process in batch groups
            all_results = []
            processed_count = 0
            
            for batch_start in range(0, total_products, batch_size):
                batch_end = min(batch_start + batch_size, total_products)
                batch_group = products[batch_start:batch_end]
                
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_products + batch_size - 1) // batch_size
                
                self.logger.info(f"Processing batch group {batch_num}/{total_batches} "
                               f"(records {batch_group[0][0]} to {batch_group[-1][0]})")
                
                # Process batch group
                batch_results = self.process_batch_group(batch_group)
                all_results.extend(batch_results)
                
                processed_count += len(batch_group)
                progress = (processed_count / total_products) * 100
                self.logger.info(f"Progress: {processed_count}/{total_products} ({progress:.1f}%)")
                
                # Write intermediate results (append mode for resume capability)
                self._append_results_to_csv(batch_results, output_file, 
                                          write_header=(batch_start == 0))
                
                # Add delay between batch groups
                if batch_end < total_products:
                    time.sleep(delay)
            
            self.logger.info(f"Batch processing completed! Results saved to {output_file}")
            self.logger.info(f"Total records processed: {len(all_results)}")
            self.logger.info(f"Efficiency gained: ~{((total_products - total_api_calls) / total_products * 100):.1f}% fewer API calls")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_file}")
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            raise Exception(f"Error processing CSV: {str(e)}")
    
    def _read_csv_range(self, input_file: str, start_record: Optional[int] = None, 
                       end_record: Optional[int] = None) -> List[Tuple[int, Dict[str, str]]]:
        """Read CSV file with optional record range filtering."""
        products = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader, 1):
                # Apply record range filtering
                if start_record and i < start_record:
                    continue
                if end_record and i > end_record:
                    break
                    
                products.append((i, row))
        
        return products
    
    def _append_results_to_csv(self, results: List[Dict[str, str]], output_file: str, 
                              write_header: bool = False) -> None:
        """Append results to output CSV file for batch processing."""
        if not results:
            return
        
        # Define output columns in the specified order
        fieldnames = [
            'record_index',
            'original_product_name',
            'original_upc_number', 
            'UPC',
            'product_name',
            'brand',
            'product_type',
            'color_desc',
            'dimension',
            'pc_count',
            'capacity',
            'configuration',
            'key_feature'
        ]
        
        # Thread-safe file writing
        with self.write_lock:
            file_mode = 'w' if write_header else 'a'
            with open(output_file, file_mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(results)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Product Analyzer - Optimized for multiple products per API request",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch processing (5 products per request)
  python batch_product_analyzer.py input.csv output.csv --products-per-request 5
  
  # Optimized batch processing
  python batch_product_analyzer.py input.csv output.csv --products-per-request 5 --batch-size 50 --threads 10
  
  # Large dataset processing
  python batch_product_analyzer.py large_file.csv output.csv --products-per-request 8 --batch-size 200 --threads 20
  
  # Process specific record range with batching
  python batch_product_analyzer.py input.csv output.csv --products-per-request 5 --start-record 1000 --end-record 2000
        """
    )
    
    parser.add_argument('input_file', help='Input CSV file with product_name and upc_number columns')
    parser.add_argument('output_file', help='Output CSV file for results')
    
    parser.add_argument('--products-per-request', type=int, default=5,
                       help='Number of products to analyze per API request (default: 5, max: 10)')
    parser.add_argument('--threads', type=int, default=10, 
                       help='Number of concurrent threads (default: 10)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of records to process per batch group (default: 50)')
    parser.add_argument('--start-record', type=int, 
                       help='Start processing from this record number (1-based)')
    parser.add_argument('--end-record', type=int,
                       help='Stop processing at this record number (1-based, inclusive)')
    parser.add_argument('--delay', type=float, default=0.50,
                       help='Delay between batch groups in seconds (default: 0.05)')
    parser.add_argument('--api-key', type=str,
                       help='OpenAI API key (if not set in environment)')
    
    return parser.parse_args()


def main():
    """Main function to run the batch product analyzer."""
    args = parse_arguments()
    
    try:
        # Validate products_per_request
        if args.products_per_request < 1 or args.products_per_request > 10:
            print("Warning: products_per_request should be between 1 and 10. Adjusting to valid range.")
            args.products_per_request = max(1, min(args.products_per_request, 10))
        
        # Initialize batch analyzer with specified parameters
        analyzer = BatchProductAnalyzer(
            api_key=args.api_key,
            max_workers=args.threads,
            products_per_request=args.products_per_request
        )
        
        # Log configuration
        analyzer.logger.info("=== Batch Product Analyzer Started ===")
        analyzer.logger.info(f"Input file: {args.input_file}")
        analyzer.logger.info(f"Output file: {args.output_file}")
        analyzer.logger.info(f"Products per API request: {args.products_per_request}")
        analyzer.logger.info(f"Threads: {args.threads}")
        analyzer.logger.info(f"Batch size: {args.batch_size}")
        analyzer.logger.info(f"Record range: {args.start_record or 'start'} to {args.end_record or 'end'}")
        analyzer.logger.info(f"Delay between batches: {args.delay}s")
        
        # Process CSV with all parameters
        analyzer.process_csv(
            input_file=args.input_file,
            output_file=args.output_file,
            batch_size=args.batch_size,
            start_record=args.start_record,
            end_record=args.end_record,
            delay=args.delay
        )
        
        analyzer.logger.info("=== Batch product analysis completed successfully! ===")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
