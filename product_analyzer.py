#!/usr/bin/env python3
"""
-------------------------------------------------------
how to run:

pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the analyzer (basic usage)
python product_analyzer.py sample_input.csv results.csv

# Advanced usage with multithreading and batching in mac
python product_analyzer.py input.csv output.csv --threads 10 --batch-size 100 --start-record 1000 --end-record 2000 --api-key=sk-proj-1234567890

# Advanced usage with multithreading and batching in windows
..\venv\Scripts\python product_analyzer.py input.csv output.csv --threads 10 --batch-size 100 --start-record 1000 --end-record 2000 --api-key=sk-proj-1234567890

# Resume processing from a specific record
venv/Scripts/python product_analyzer.py input.csv output.csv --start-record 5000 --threads 5

model configured is gpt-4
-------------------------------------------------------

Product Analyzer Script

This script reads product names and UPC numbers from a CSV file,
sends them to OpenAI API for analysis, and saves the results to an output CSV.

Features:
- Multithreading for concurrent API calls
- Batching for efficient processing of large datasets
- Start/end record parameters for partial processing
- Progress tracking and logging
- Resume capability

Required environment variable:
- OPENAI_API_KEY: Your OpenAI API key

Usage:
    python product_analyzer.py input.csv output.csv [options]
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


class ProductAnalyzer:
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 5):
        """Initialize the ProductAnalyzer with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.prompt_template = self._load_prompt_template()
        self.max_workers = max_workers
        self.write_lock = Lock()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('product_analyzer.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the text file."""
        script_dir = Path(__file__).parent
        prompt_file = script_dir / "product_analysis_prompt.txt"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template file not found: {prompt_file}")
    
    def analyze_product(self, product_name: str, upc_number: str, record_index: int = 0) -> Dict[str, str]:
        """
        Analyze a single product using OpenAI API.
        
        Args:
            product_name: Name of the product
            upc_number: UPC number of the product
            record_index: Index of the record for logging
            
        Returns:
            Dictionary containing the analyzed product information
        """
        try:
            # Format the prompt with product information
            prompt = self.prompt_template.format(
                product_name=product_name,
                upc_number=upc_number
            )
            
            # Call OpenAI API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful product analysis assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.1
                    )
                    break
                except Exception as api_error:
                    if attempt == max_retries - 1:
                        raise api_error
                    self.logger.warning(f"API call failed for record {record_index}, attempt {attempt + 1}: {api_error}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Extract the response content
            response_content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(response_content)
                # Add original input data and record index
                result['original_product_name'] = product_name
                result['original_upc_number'] = upc_number
                result['record_index'] = record_index
                return result
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse JSON response for record {record_index} ({product_name}). Raw response: {response_content}")
                return self._create_fallback_result(product_name, upc_number, record_index)
                
        except Exception as e:
            self.logger.error(f"Error analyzing product at record {record_index} ({product_name}): {str(e)}")
            return self._create_fallback_result(product_name, upc_number, record_index)
    
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
    
    def process_batch(self, batch: List[Tuple[int, Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Process a batch of products using multithreading.
        
        Args:
            batch: List of tuples (record_index, product_dict)
            
        Returns:
            List of analyzed product results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_record = {
                executor.submit(
                    self.analyze_product, 
                    product['product_name'], 
                    product['upc_number'],
                    record_index
                ): record_index 
                for record_index, product in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_record):
                record_index = future_to_record[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Thread execution failed for record {record_index}: {e}")
                    # Create fallback result
                    product = next(p for r, p in batch if r == record_index)
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
                   delay: float = 0.1) -> None:
        """
        Process a CSV file containing product names and UPC numbers with batching and multithreading.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            batch_size: Number of records to process in each batch
            start_record: Start processing from this record number (1-based)
            end_record: Stop processing at this record number (inclusive, 1-based)
            delay: Delay between batches to avoid rate limiting
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
            self.logger.info(f"Using {self.max_workers} threads with batch size {batch_size}")
            
            # Process in batches
            all_results = []
            processed_count = 0
            
            for batch_start in range(0, total_products, batch_size):
                batch_end = min(batch_start + batch_size, total_products)
                batch = products[batch_start:batch_end]
                
                batch_num = (batch_start // batch_size) + 1
                total_batches = (total_products + batch_size - 1) // batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} "
                               f"(records {batch[0][0]} to {batch[-1][0]})")
                
                # Process batch
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)
                
                processed_count += len(batch)
                progress = (processed_count / total_products) * 100
                self.logger.info(f"Progress: {processed_count}/{total_products} ({progress:.1f}%)")
                
                # Write intermediate results (append mode for resume capability)
                self._append_results_to_csv(batch_results, output_file, 
                                          write_header=(batch_start == 0))
                
                # Add delay between batches
                if batch_end < total_products:
                    time.sleep(delay)
            
            self.logger.info(f"Processing completed! Results saved to {output_file}")
            self.logger.info(f"Total records processed: {len(all_results)}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_file}")
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            raise Exception(f"Error processing CSV: {str(e)}")
    
    def _read_csv_range(self, input_file: str, start_record: Optional[int] = None, 
                       end_record: Optional[int] = None) -> List[Tuple[int, Dict[str, str]]]:
        """
        Read CSV file with optional record range filtering.
        
        Args:
            input_file: Path to input CSV file
            start_record: Start record number (1-based, inclusive)
            end_record: End record number (1-based, inclusive)
            
        Returns:
            List of tuples (record_index, product_dict)
        """
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
    
    def _write_output_csv(self, results: List[Dict[str, str]], output_file: str) -> None:
        """Write results to output CSV file."""
        if not results:
            raise ValueError("No results to write")
        
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
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
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
        description="Product Analyzer with OpenAI API - Enhanced for large datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python product_analyzer.py input.csv output.csv
  
  # With multithreading and batching
  python product_analyzer.py input.csv output.csv --threads 10 --batch-size 100
  
  # Process specific record range
  python product_analyzer.py input.csv output.csv --start-record 1000 --end-record 2000
  
  # Resume processing from a specific record
  python product_analyzer.py input.csv output.csv --start-record 5000 --threads 5
  
  # Large dataset processing
  python product_analyzer.py million_records.csv results.csv --threads 20 --batch-size 200 --delay 0.05
        """
    )
    
    parser.add_argument('input_file', help='Input CSV file with product_name and upc_number columns')
    parser.add_argument('output_file', help='Output CSV file for results')
    
    parser.add_argument('--threads', type=int, default=5, 
                       help='Number of concurrent threads (default: 5)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of records to process per batch (default: 50)')
    parser.add_argument('--start-record', type=int, 
                       help='Start processing from this record number (1-based)')
    parser.add_argument('--end-record', type=int,
                       help='Stop processing at this record number (1-based, inclusive)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between batches in seconds (default: 0.1)')
    parser.add_argument('--api-key', type=str,
                       help='OpenAI API key (if not set in environment)')
    
    return parser.parse_args()


def main():
    """Main function to run the product analyzer."""
    args = parse_arguments()
    
    try:
        # Initialize analyzer with specified parameters
        analyzer = ProductAnalyzer(
            api_key=args.api_key,
            max_workers=args.threads
        )
        
        # Log configuration
        analyzer.logger.info("=== Product Analyzer Started ===")
        analyzer.logger.info(f"Input file: {args.input_file}")
        analyzer.logger.info(f"Output file: {args.output_file}")
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
        
        analyzer.logger.info("=== Product analysis completed successfully! ===")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
