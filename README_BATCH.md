# Batch Product Analyzer - Separate Script

## 🚀 Overview

This is a **dedicated batch processing script** (`batch_product_analyzer.py`) designed specifically for processing multiple products per OpenAI API request. Unlike the original script, this focuses entirely on batch efficiency.

## 📁 Files

- **`batch_product_analyzer.py`** - Main batch processing script
- **`test_batch_script.py`** - Test script for the batch analyzer
- **`test_batch_input.csv`** - Sample test data (15 products)
- **`batch_product_analysis_prompt.txt`** - Batch prompt template (auto-generated)
- **`README_BATCH.md`** - This documentation

## ⚡ Key Features

### 🎯 **Optimized for Batching**
- **Default**: 5 products per API request
- **Configurable**: 1-10 products per request
- **Intelligent**: Auto-creates prompt template if missing

### 📊 **Efficiency Gains**
- **80% fewer API calls** (5 products per request vs 1)
- **Faster processing** for large datasets
- **Lower costs** due to reduced API usage

### 🔧 **Built-in Optimizations**
- Multithreading for concurrent batch processing
- Progress tracking and logging
- Resume capability with intermediate saves
- Error handling with fallback results

## 🛠️ Installation & Setup

### 1. Requirements
```bash
pip install openai
```

### 2. Set API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Verify Setup
```bash
python test_batch_script.py
```

## 📋 Usage

### **Your Specific Scenario (5 products per request, 50 batch size)**
```bash
python batch_product_analyzer.py input.csv output.csv --products-per-request 5 --batch-size 50 --threads 10
```

### **Basic Batch Processing**
```bash
python batch_product_analyzer.py input.csv output.csv --products-per-request 5
```

### **Large Dataset Processing**
```bash
python batch_product_analyzer.py large_file.csv output.csv --products-per-request 8 --batch-size 200 --threads 20
```

### **Process Specific Range**
```bash
python batch_product_analyzer.py input.csv output.csv --products-per-request 5 --start-record 1000 --end-record 2000
```

## 📊 Command Line Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--products-per-request` | Products per API call | 5 | 1-10 |
| `--batch-size` | Records per processing group | 50 | Any |
| `--threads` | Concurrent threads | 10 | Any |
| `--delay` | Delay between groups (seconds) | 0.05 | Any |
| `--start-record` | Start from record (1-based) | 1 | Any |
| `--end-record` | End at record (1-based) | Last | Any |
| `--api-key` | OpenAI API key | ENV var | Any |

## 💡 How Batch Processing Works

### Traditional Approach (1 product per request)
```
Input: 15 products
├── API Call 1: Product 1
├── API Call 2: Product 2
├── ...
└── API Call 15: Product 15
Total: 15 API calls
```

### Batch Approach (5 products per request)
```
Input: 15 products
├── API Call 1: Products 1-5
├── API Call 2: Products 6-10
└── API Call 3: Products 11-15
Total: 3 API calls (80% reduction!)
```

## 📈 Performance Comparison

### Example: 1000 Products
| Approach | API Calls | Estimated Time | Cost Impact |
|----------|-----------|----------------|-------------|
| Traditional (1 per request) | 1000 | 10-15 minutes | 100% |
| Batch (5 per request) | 200 | 3-5 minutes | 20% |
| Batch (10 per request) | 100 | 2-3 minutes | 10% |

## 🎯 Recommended Settings

### By Dataset Size
| Products | Products/Request | Batch Size | Threads | Use Case |
|----------|------------------|------------|---------|----------|
| < 100 | 3-5 | 20-50 | 5-10 | Small datasets |
| 100-1000 | 5-8 | 50-100 | 10-15 | Medium datasets |
| 1000-10000 | 8-10 | 100-200 | 15-20 | Large datasets |
| > 10000 | 10 | 200-500 | 20+ | Very large datasets |

### Your Scenario (5 products per request, 50 batch size)
Perfect for:
- **Medium to large datasets** (500-5000 products)
- **Balanced efficiency** and API rate limit compliance
- **Good performance** without overwhelming the API

## 🧪 Testing

### Run the Test Script
```bash
python test_batch_script.py
```

This will:
1. ✅ Check requirements (API key, files)
2. 🚀 Run batch processing with your scenario
3. 📊 Show efficiency comparison
4. 📝 Display usage examples

### Expected Output
```
✅ Batch processing completed successfully!
⏱️  Duration: 15.32 seconds
📄 Output file: test_batch_output.csv
📊 Total rows: 16 (including header)
📈 Products processed: 15

💰 Efficiency gain: 66.7% fewer API calls
🚀 Speed improvement: ~3.0x faster
```

## 📝 Output Format

Same CSV format as the original script:
```csv
record_index,original_product_name,original_upc_number,UPC,product_name,brand,product_type,color_desc,dimension,pc_count,capacity,configuration,key_feature
1,Apple iPhone 14 Pro Max 256GB Space Black,123456789012,123456789012,iPhone 14 Pro Max,Apple,Electronics,Space Black,6.7 inch,1,256GB,Pro Max,A16 Bionic chip
```

## 🔍 Logging

The script creates detailed logs in `batch_product_analyzer.log`:
- Processing progress
- API call efficiency metrics
- Error handling and fallbacks
- Performance statistics

## ⚠️ Important Notes

### 1. **API Rate Limits**
- OpenAI has rate limits (requests per minute)
- Batch processing reduces request count
- Adjust `--delay` if hitting limits

### 2. **Token Limits**
- Each API call is limited by tokens (~4000 for GPT-4)
- More products per request = more tokens
- Recommended max: 10 products per request

### 3. **Error Handling**
- Failed batches generate fallback results
- Processing continues with remaining batches
- All errors logged for review

## 🚦 Getting Started

1. **Quick Test**:
   ```bash
   python test_batch_script.py
   ```

2. **Your Scenario**:
   ```bash
   python batch_product_analyzer.py your_file.csv results.csv --products-per-request 5 --batch-size 50 --threads 10
   ```

3. **Monitor Progress**:
   ```bash
   tail -f batch_product_analyzer.log
   ```

## 🆚 vs Original Script

| Feature | Original Script | Batch Script |
|---------|----------------|--------------|
| **Focus** | General purpose | Batch optimization |
| **Default behavior** | 1 product/request | 5 products/request |
| **Complexity** | Higher (backward compatibility) | Simpler (batch-focused) |
| **Performance** | Standard | Optimized for batching |
| **Use case** | Mixed usage | Large datasets |

windows run
```bash
..\venv\Scripts\python batch_product_analyzer.py sample_input.csv output.csv --products-per-request 5 --threads 10 --batch-size 100 --start-record 1 --end-record 50 --api-key=your_key
..\venv\Scripts\python batch_product_analyzer.py sample_input.csv output.csv --products-per-request 5 --threads 2 --batch-size 50 --start-record 1 --end-record 1000 --api-key=your_key
```

The batch script is **purpose-built** for your use case: efficient processing of large datasets with multiple products per API request!
