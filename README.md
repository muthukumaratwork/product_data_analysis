# Enhanced Product Analyzer with OpenAI GPT-4 API

A high-performance product analysis tool designed to handle millions of records using multithreading, batching, and smart record processing. Uses GPT-4 for superior analysis quality.

## 🚀 Key Features

- **Multithreading**: Concurrent API calls for maximum throughput
- **Batching**: Memory-efficient processing of large datasets  
- **Record Range Processing**: Start/end record parameters for partial processing
- **Resume Capability**: Continue processing from any record
- **Progress Tracking**: Detailed logging and progress monitoring
- **Thread-Safe Operations**: Safe concurrent file writing
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Graceful fallback for failed API calls

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## 🔧 Usage

### Basic Usage
```bash
python product_analyzer.py input.csv output.csv
```

### Enhanced Usage for Large Datasets

**With multithreading and batching:**
```bash
python product_analyzer.py input.csv output.csv --threads 10 --batch-size 100
```

**Process specific record range:**
```bash
python product_analyzer.py input.csv output.csv --start-record 1000 --end-record 2000
```

**Resume processing from a specific record:**
```bash
python product_analyzer.py input.csv output.csv --start-record 5000 --threads 5
```

**High-performance processing for millions of records:**
```bash
python product_analyzer.py million_records.csv results.csv --threads 20 --batch-size 200 --delay 0.05
```

## 📊 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--threads` | Number of concurrent threads | 5 |
| `--batch-size` | Records to process per batch | 50 |
| `--start-record` | Start from this record (1-based) | 1 |
| `--end-record` | End at this record (1-based) | last |
| `--delay` | Delay between batches (seconds) | 0.1 |
| `--api-key` | OpenAI API key (if not in env) | - |

## 📁 Input CSV Format

Your input CSV must contain:
```csv
product_name,upc_number
"Apple iPhone 14 Pro 128GB Space Black","194253399616"
"Samsung 65-inch 4K Smart TV","887276647825"
```

## 📄 Output CSV Format

The output includes:
- `record_index`: Original record position
- `original_product_name`: Input product name
- `original_upc_number`: Input UPC number
- `UPC`: Validated UPC number
- `product_name`: Standardized product name
- `brand`: Product brand
- `product_type`: Category/type
- `color_desc`: Color description
- `dimension`: Physical dimensions
- `pc_count`: Number of pieces/units
- `capacity`: Storage capacity or volume
- `configuration`: Product configuration
- `key_feature`: Main distinguishing features

## 📈 Performance Optimization

### For Million+ Record Datasets:

1. **Increase thread count** (but watch API rate limits):
   ```bash
   --threads 20
   ```

2. **Larger batch sizes** for better throughput:
   ```bash
   --batch-size 200
   ```

3. **Reduce delay** between batches:
   ```bash
   --delay 0.05
   ```

4. **Process in chunks** to avoid memory issues:
   ```bash
   # Process first 100K records
   python product_analyzer.py huge_file.csv part1.csv --end-record 100000
   
   # Process next 100K records  
   python product_analyzer.py huge_file.csv part2.csv --start-record 100001 --end-record 200000
   ```

## 📝 Logging

The script generates detailed logs in `product_analyzer.log`:
- Processing progress
- Error details
- Performance metrics
- API call status

## 🛠️ Error Handling

- **API Failures**: Automatic retry with exponential backoff
- **Rate Limiting**: Configurable delays between batches
- **Memory Management**: Batch processing prevents memory overflow
- **Interrupted Processing**: Resume from any record position

## 📋 Example Scenarios

### Scenario 1: Quick Analysis (< 1000 records)
```bash
python product_analyzer.py small_dataset.csv results.csv
```

### Scenario 2: Medium Dataset (1K - 100K records)
```bash
python product_analyzer.py medium_dataset.csv results.csv --threads 10 --batch-size 100
```

### Scenario 3: Large Dataset (100K - 1M records)
```bash
python product_analyzer.py large_dataset.csv results.csv --threads 15 --batch-size 150 --delay 0.05
```

### Scenario 4: Million+ Records
```bash
# Process in chunks
python product_analyzer.py million_records.csv chunk1.csv --end-record 500000 --threads 20 --batch-size 200
python product_analyzer.py million_records.csv chunk2.csv --start-record 500001 --threads 20 --batch-size 200
```

## 🔄 Resume Processing

If processing is interrupted, you can resume from where it left off:

```bash
# Check the last processed record in your output file
# Then resume from the next record
python product_analyzer.py input.csv output.csv --start-record 12500 --threads 10
```

## ⚠️ Important Notes

- **API Rate Limits**: OpenAI has rate limits. Adjust `--threads` and `--delay` accordingly
- **Memory Usage**: Large batch sizes use more memory but improve performance
- **Output File**: Results are appended in real-time for resume capability
- **Thread Safety**: All file operations are thread-safe

## 🆘 Troubleshooting

**"Rate limit exceeded"**: Reduce `--threads` or increase `--delay`

**Memory issues**: Reduce `--batch-size`

**Slow processing**: Increase `--threads` and reduce `--delay`

**API errors**: Check your OpenAI API key and quota

---
windows run
```bash
..\venv\Scripts\python product_analyzer.py sample_input.csv output.csv --threads 10 --batch-size 100 --start-record 1 --end-record 50 --api-key=your_key
```


This enhanced version can efficiently process millions of records with optimal performance and reliability.
