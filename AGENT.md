# AI Agent & Developer Guide - German Sentiment Analysis System

## 🎯 System Architecture

### Core Components
```
src/sentiment_analysis/
├── __init__.py          # Package exports
└── analyzer.py          # Main GermanFeedbackAnalyzer class

tests/
├── test_analyzer.py     # Core functionality tests
└── test_data_samples.py # Test fixtures

main.py                  # Entry point
run_tests.py            # Test runner
```

### Technology Stack Tiers

**Tier 1 - Minimal (Always Works)**
```python
pandas, numpy
→ Basic clustering + rule-based sentiment
```

**Tier 2 - Standard**
```python
+ scikit-learn, nltk, matplotlib, openpyxl, vaderSentiment
→ TF-IDF clustering + VADER sentiment + visualizations
```

**Tier 3 - Advanced (Recommended)**
```python
+ sentence-transformers, umap-learn, hdbscan, seaborn, transformers
→ Neural embeddings + HDBSCAN + German BERT sentiment
```

## 🔧 Key Technical Implementation

### Sentiment Analysis Priority
1. **German BERT** (`oliverguhr/german-sentiment-bert`) - Most accurate for German
2. **Enhanced VADER** - With German word mappings
3. **Rule-based German** - Fallback with German lexicon

### Clustering Strategy
```python
# Neural embeddings: Small, precise clusters
HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.1)

# TF-IDF: Larger clusters to avoid noise
HDBSCAN(min_cluster_size=30, min_samples=10)
```

### German Language Optimizations
- **Column Priority**: `feedback` → `message` → `message_de`
- **Stopwords**: Extended German + English mixed content
- **Cluster Labels**: Business-contextual German terminology
- **Sentiment Boost**: German positive/negative word mappings

## 🚀 Development Workflow

### Quick Setup
```bash
git clone <repo>
cd sentiment-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Tests
```bash
# All tests
python run_tests.py

# Specific test
python -m pytest tests/test_analyzer.py::TestGermanFeedbackAnalyzer::test_sentiment_analysis -v
```

### Code Structure
```python
from src.sentiment_analysis import GermanFeedbackAnalyzer

# Initialize with auto-capability detection
analyzer = GermanFeedbackAnalyzer('data/feedback.xlsx')

# Check available features
print(analyzer.capabilities)
# {'sklearn': True, 'transformers': True, 'hdbscan': True, ...}

# Run full pipeline
results_df, summary_df = analyzer.run_analysis()
```

## 🔍 Debugging Guide

### Common Issues & Solutions

**1. All Sentiment Neutral**
```python
# Check if German BERT is working
from transformers import pipeline
classifier = pipeline('sentiment-analysis', model='oliverguhr/german-sentiment-bert')
print(classifier('Das ist ausgezeichnet!'))
# Should return: [{'label': 'positive', 'score': 0.996}]
```

**2. Clustering Issues**
```python
# Check embedding quality
embeddings = analyzer.create_embeddings(['test text'])
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding range: {embeddings.min():.3f} to {embeddings.max():.3f}")
```

**3. Missing Dependencies**
```python
# Check capability detection
analyzer = GermanFeedbackAnalyzer()
missing = [k for k, v in analyzer.capabilities.items() if not v]
print(f"Missing packages: {missing}")
```

### Log Analysis
```bash
# Real-time monitoring
tail -f feedback_analysis.log

# Error filtering
grep "ERROR\|WARNING" feedback_analysis.log
```

## 📊 Data Pipeline Details

### Input Processing
```python
# Column mapping priority
column_mapping = {
    'feedback': 'message',      # Primary
    'message_de': 'message',    # German preferred
    'nachricht': 'message',     # German alternative
    'datum': 'date',           # German date
    'created_at': 'date'       # Alternative date
}
```

### Sentiment Analysis Implementation
```python
def analyze_sentiment(self, texts):
    # Priority order:
    try:
        return self._german_bert_sentiment_analysis(texts)
    except ImportError:
        if self.capabilities['vader']:
            return self._enhanced_vader_sentiment_analysis(texts)
        else:
            return self._german_sentiment_analysis(texts)
```

### Cluster Labeling Logic
```python
# Business domain mapping
term_mapping = {
    'support': 'Support', 'login': 'Anmeldung', 
    'shipping': 'Versand', 'billing': 'Abrechnung',
    'dashboard': 'Dashboard', 'performance': 'Performance'
}

# Contextual analysis
if 'support' in sample_text and 'callback' in sample_text:
    return 'Support Rückruf'
elif 'login' in sample_text or 'timeout' in sample_text:
    return 'Anmelde-Probleme'
```

## 🧪 Testing Strategy

### Test Coverage
- **Unit Tests**: Individual methods (sentiment, clustering, labeling)
- **Integration Tests**: Full pipeline with sample data
- **Edge Cases**: Empty data, missing columns, mixed languages

### Test Data Samples
```python
# German positive
"Der Kundensupport war ausgezeichnet und sehr hilfreich!"

# German negative  
"Die App-Oberfläche ist schlecht und unübersichtlich."

# Mixed language
"Der Service war excellent, but shipping took too long."
```

### Performance Benchmarks
```python
# Expected performance (500 messages)
- Loading: < 1 second
- Embeddings: 3-5 seconds (neural), < 1 second (TF-IDF)
- Clustering: 8-10 seconds (HDBSCAN), 2-3 seconds (DBSCAN)
- Sentiment: 5-7 seconds (BERT), < 1 second (VADER)
- Total: ~20 seconds (full stack), ~5 seconds (minimal)
```

## 🔄 Extension Points

### Adding New Languages
```python
def _get_spanish_stopwords(self):
    return {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'}

def _create_spanish_label(self, terms, texts):
    # Spanish business term mapping
    pass
```

### Custom Clustering Algorithms
```python
def _custom_clustering(self, X):
    # Implement new clustering method
    # Must return cluster labels array
    pass

# Add to cluster_texts() method with capability check
```

### New Visualization Types
```python
def create_custom_viz(self, df):
    if not self.capabilities['matplotlib']:
        return
    
    # Custom chart implementation
    plt.figure(figsize=(12, 8))
    # ... plotting code
    plt.savefig(self.output_dir / 'custom_chart.png')
```

## 🚀 Production Deployment

### Docker Setup
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY main.py .

CMD ["python", "main.py"]
```

### Environment Configuration
```bash
# Required
export INPUT_FILE="data/feedback.xlsx"

# Optional
export OUTPUT_DIR="results/"
export LOG_LEVEL="INFO"
export CUDA_VISIBLE_DEVICES="0"  # GPU acceleration
```

### Monitoring & Alerting
```python
# Key metrics to monitor
- Processing time per message
- Memory usage during analysis
- Cluster count (should be 10-50 for good segmentation)
- Sentiment distribution (balanced is healthy)
- Error rate in sentiment analysis
```

## 🔒 Security & Performance

### Input Validation
```python
# File validation
if not file_path.suffix.lower() in ['.csv', '.xlsx']:
    raise ValueError("Unsupported file format")

# Data validation  
df = df[df['message'].str.len() < 10000]  # Prevent memory issues
df = df[df['message'].str.len() > 5]      # Filter noise
```

### Memory Management
```python
# Large dataset handling
if len(texts) > 10000:
    # Process in batches
    batch_size = 1000
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Process batch
```

### Error Recovery
```python
# Graceful degradation
try:
    return self._advanced_method()
except Exception as e:
    logger.warning(f"Advanced method failed: {e}")
    return self._fallback_method()
```

## 📈 Performance Optimization

### GPU Acceleration
```python
# Automatic GPU detection
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
```

### Batch Processing
```python
# Efficient sentiment analysis
classifier = pipeline("sentiment-analysis", 
                     model="oliverguhr/german-sentiment-bert",
                     device=0 if torch.cuda.is_available() else -1)

# Process in batches instead of individual texts
results = classifier(texts, batch_size=32)
```

### Memory Optimization
```python
# Clear large objects
del embeddings, X_reduced
import gc; gc.collect()
```

---

**This system is designed for enterprise-grade German customer feedback analysis with maximum reliability, performance, and maintainability.**