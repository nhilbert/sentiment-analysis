#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
German Customer Feedback Analysis System
=======================================

A production-ready system for analyzing German customer feedback using semantic clustering
and sentiment analysis. Automatically adapts to available packages and provides fallbacks.

Features:
- Semantic clustering using neural embeddings (SentenceTransformers + HDBSCAN)
- German-focused sentiment analysis with VADER
- Automatic fallbacks for missing dependencies
- German cluster labeling and visualization
- Excel/CSV input/output support

Author: AI Assistant
Version: 1.0
Date: 2024
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

# Configure logging for production use
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feedback_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GermanFeedbackAnalyzer:
    """
    Production-ready German customer feedback analysis system.
    
    This class provides comprehensive analysis of German customer feedback including:
    - Semantic text clustering using neural embeddings
    - Sentiment analysis optimized for German text
    - Automatic visualization generation
    - Flexible input/output handling
    
    The system automatically detects available packages and provides intelligent
    fallbacks to ensure functionality even with minimal dependencies.
    """
    
    def __init__(self, input_file: str = 'data/kundenfeedback_beispiel.xlsx'):
        """
        Initialize the analyzer with configuration.
        
        Args:
            input_file (str): Path to input Excel/CSV file with customer feedback
        """
        self.input_file = input_file
        self.output_dir = Path('output')
        self.df = None
        self.cluster_meta = {}
        
        # Detect available advanced packages
        self.capabilities = self._detect_capabilities()
        
        # Log system configuration
        logger.info(f"Initialized German Feedback Analyzer")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Advanced features: {sum(self.capabilities.values())}/{len(self.capabilities)}")
        
    def _detect_capabilities(self) -> Dict[str, bool]:
        """
        Detect which advanced packages are available for enhanced functionality.
        
        Returns:
            Dict[str, bool]: Mapping of package names to availability status
        """
        capabilities = {}
        
        # Test each package individually to provide graceful degradation
        test_imports = {
            'sklearn': 'sklearn',
            'sentence_transformers': 'sentence_transformers', 
            'umap': 'umap',
            'hdbscan': 'hdbscan',
            'vader': 'vaderSentiment',
            'nltk': 'nltk',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'openpyxl': 'openpyxl'
        }
        
        for capability, module_name in test_imports.items():
            try:
                __import__(module_name)
                capabilities[capability] = True
                logger.info(f"✓ {capability} available")
            except ImportError as e:
                capabilities[capability] = False
                if capability == 'matplotlib' and 'pyparsing' in str(e):
                    logger.warning(f"✗ {capability} not available - missing pyparsing dependency")
                    logger.info("   Fix: pip install pyparsing matplotlib")
                elif capability == 'seaborn' and ('matplotlib' in str(e) or 'pyparsing' in str(e)):
                    logger.warning(f"✗ {capability} not available - missing matplotlib dependencies")
                    logger.info("   Fix: pip install pyparsing matplotlib seaborn")
                else:
                    logger.warning(f"✗ {capability} not available - using fallback")
        
        return capabilities
    
    def load_data(self) -> pd.DataFrame:
        """
        Load customer feedback data from Excel or CSV files.
        
        Returns:
            pd.DataFrame: Cleaned dataframe with feedback data
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        file_path = Path(self.input_file)
        
        # Handle different file formats
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif self.capabilities['openpyxl']:
                df = pd.read_excel(file_path)
            else:
                # Fallback: try CSV with same name
                csv_path = file_path.with_suffix('.csv')
                if csv_path.exists():
                    df = pd.read_csv(csv_path, encoding='utf-8')
                else:
                    raise ImportError("Install openpyxl for Excel support or provide CSV file")
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
        
        # Normalize column names for consistent processing
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Validate required columns
        required_columns = ['date', 'message']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Try alternative column names - prioritize German content
            column_mapping = {
                'message_de': 'message',  # Prioritize German messages
                'nachricht': 'message',
                'datum': 'date',
                'created_at': 'date'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name in missing_columns:
                    df = df.rename(columns={old_name: new_name})
                    missing_columns.remove(new_name)
                    logger.info(f"Using {old_name} as {new_name} column")
            
            if missing_columns:
                raise ValueError(f"Required columns missing: {missing_columns}")
        
        logger.info(f"Successfully loaded {len(df)} records from {self.input_file}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the feedback data.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe ready for analysis
        """
        initial_count = len(df)
        
        # Clean message text
        df['message'] = df['message'].astype(str).str.strip()
        df = df[df['message'].notna() & (df['message'] != '') & (df['message'] != 'nan')]
        
        # Process dates with error handling
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()]
        
        # Create time-based features for analysis
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        df['month_name'] = df['date'].dt.strftime('%B %Y')
        
        # Use German message if available, otherwise use message column
        if 'message_de' in df.columns:
            df['nachricht'] = df['message_de']
            logger.info("Using German messages (message_de) for analysis")
        else:
            df = df.rename(columns={'message': 'nachricht'})
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        logger.info(f"Data cleaning complete: {final_count} valid records ({removed_count} removed)")
        return df
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create text embeddings using the best available method.
        
        Priority: SentenceTransformers > TF-IDF > Basic word vectors
        
        Args:
            texts (List[str]): List of text messages to embed
            
        Returns:
            np.ndarray: Normalized text embeddings
        """
        if self.capabilities['sentence_transformers']:
            logger.info("Creating neural embeddings with SentenceTransformers")
            return self._create_neural_embeddings(texts)
        elif self.capabilities['sklearn']:
            logger.info("Creating TF-IDF embeddings")
            return self._create_tfidf_embeddings(texts)
        else:
            logger.info("Creating basic word frequency embeddings")
            return self._create_basic_embeddings(texts)
    
    def _create_neural_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create high-quality neural embeddings using SentenceTransformers."""
        from sentence_transformers import SentenceTransformer
        
        # Use multilingual model that works well with German
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        
        # Normalize embeddings for better clustering
        if self.capabilities['sklearn']:
            from sklearn.preprocessing import normalize
            return normalize(embeddings, norm="l2")
        else:
            # Manual L2 normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / (norms + 1e-8)
    
    def _create_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF embeddings with German stopwords."""
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.preprocessing import normalize
        
        # Get German stopwords
        stopwords = self._get_german_stopwords()
        
        # Create TF-IDF vectors
        vectorizer = CountVectorizer(
            max_features=5000, 
            ngram_range=(1,2), 
            stop_words=list(stopwords)
        )
        X_counts = vectorizer.fit_transform(texts)
        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X_counts)
        
        return normalize(X_tfidf, norm="l2")
    
    def _create_basic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback: Create basic word frequency vectors."""
        # Simple tokenization and vocabulary building
        vocab = set()
        for text in texts:
            words = text.lower().split()
            vocab.update(words)
        
        # Limit vocabulary size for performance
        vocab = sorted(list(vocab))[:5000]
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # Create frequency matrix
        vectors = np.zeros((len(texts), len(vocab)))
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in word_to_idx:
                    vectors[i, word_to_idx[word]] += 1
        
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)
    
    def reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction for better clustering.
        
        Args:
            X (np.ndarray): High-dimensional embeddings
            
        Returns:
            np.ndarray: Reduced-dimension embeddings
        """
        if self.capabilities['umap']:
            logger.info("Applying UMAP dimensionality reduction")
            import umap
            
            # Optimize UMAP parameters for clustering
            reducer = umap.UMAP(
                n_components=10, 
                n_neighbors=15, 
                min_dist=0.0, 
                metric="euclidean", 
                random_state=42
            )
            return reducer.fit_transform(X)
        else:
            logger.info("Skipping dimensionality reduction")
            return X.toarray() if hasattr(X, 'toarray') else X
    
    def cluster_texts(self, X: np.ndarray) -> np.ndarray:
        """
        Perform semantic clustering of text embeddings.
        
        Args:
            X (np.ndarray): Text embeddings
            
        Returns:
            np.ndarray: Cluster labels for each text
        """
        if self.capabilities['hdbscan']:
            logger.info("Performing HDBSCAN clustering")
            return self._hdbscan_clustering(X)
        elif self.capabilities['sklearn']:
            logger.info("Performing DBSCAN clustering")
            return self._dbscan_clustering(X)
        else:
            logger.info("Performing basic similarity clustering")
            return self._basic_clustering(X)
    
    def _hdbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """Advanced clustering with HDBSCAN - optimized for semantic separation."""
        import hdbscan
        
        # Adjust parameters based on embedding quality
        if self.capabilities['sentence_transformers']:
            # With neural embeddings: smaller clusters for better topic separation
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=2,
                metric="euclidean",
                cluster_selection_epsilon=0.1
            )
        else:
            # With TF-IDF: larger clusters to avoid noise
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=30,
                min_samples=10,
                metric="euclidean"
            )
        
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"HDBSCAN found {n_clusters} semantic clusters")
        
        return labels
    
    def _dbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """Intermediate clustering with DBSCAN."""
        from sklearn.cluster import DBSCAN
        
        clusterer = DBSCAN(eps=0.8, min_samples=10, metric="euclidean")
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"DBSCAN found {n_clusters} clusters")
        
        return labels
    
    def _basic_clustering(self, X: np.ndarray) -> np.ndarray:
        """Fallback clustering using cosine similarity."""
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        threshold = 0.7
        cluster_id = 0
        
        for i in range(n_samples):
            if labels[i] != -1:
                continue
            
            # Find similar points using cosine similarity
            similarities = np.dot(X, X[i])
            similar_points = np.where(similarities > threshold)[0]
            
            if len(similar_points) >= 10:  # min_samples
                labels[similar_points] = cluster_id
                cluster_id += 1
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Basic clustering found {n_clusters} clusters")
        
        return labels
    
    def generate_cluster_labels(self, texts: List[str], labels: np.ndarray) -> Dict[int, Dict]:
        """
        Generate meaningful German labels for discovered clusters.
        
        Args:
            texts (List[str]): Original text messages
            labels (np.ndarray): Cluster assignments
            
        Returns:
            Dict[int, Dict]: Cluster metadata with German labels
        """
        # Group texts by cluster
        clusters = {}
        for text, label in zip(texts, labels):
            clusters.setdefault(label, []).append(text)
        
        if self.capabilities['sklearn']:
            logger.info("Generating German cluster labels with TF-IDF analysis")
            return self._advanced_german_labeling(clusters)
        else:
            logger.info("Generating German cluster labels with basic analysis")
            return self._basic_german_labeling(clusters)
    
    def _advanced_german_labeling(self, clusters: Dict[int, List[str]]) -> Dict[int, Dict]:
        """Generate German cluster labels using TF-IDF analysis."""
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        
        # Get German stopwords
        stopwords = self._get_german_stopwords()
        
        # Create documents per cluster
        documents = []
        cluster_ids = []
        for cluster_id, texts in clusters.items():
            if cluster_id != -1:  # Skip noise cluster
                documents.append(' '.join(texts))
                cluster_ids.append(cluster_id)
        
        if not documents:
            return {}
        
        # TF-IDF analysis optimized for German text
        vectorizer = CountVectorizer(
            ngram_range=(1,2), 
            stop_words=list(stopwords), 
            max_features=5000, 
            min_df=1
        )
        counts = vectorizer.fit_transform(documents)
        tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
        tfidf_matrix = tfidf.fit_transform(counts)
        vocab = vectorizer.get_feature_names_out()
        
        # Extract top German terms per cluster
        cluster_meta = {}
        for i, cluster_id in enumerate(cluster_ids):
            row = tfidf_matrix[i].toarray().ravel()
            top_indices = np.argsort(-row)[:8]
            top_terms = [vocab[idx] for idx in top_indices if row[idx] > 0]
            
            # Create contextual German label
            german_label = self._create_contextual_german_label(top_terms, clusters[cluster_id])
            
            cluster_meta[cluster_id] = {
                'label': german_label,
                'keywords': top_terms,
                'size': len(clusters[cluster_id])
            }
        
        # Add noise cluster
        if -1 in clusters:
            cluster_meta[-1] = {
                'label': 'Rauschen',
                'keywords': [],
                'size': len(clusters[-1])
            }
        
        return cluster_meta
    
    def _basic_german_labeling(self, clusters: Dict[int, List[str]]) -> Dict[int, Dict]:
        """Generate German cluster labels using basic word frequency."""
        stopwords = self._get_german_stopwords()
        cluster_meta = {}
        
        for cluster_id, texts in clusters.items():
            if cluster_id == -1:
                cluster_meta[cluster_id] = {
                    'label': 'Rauschen',
                    'keywords': [],
                    'size': len(texts)
                }
                continue
            
            # Count word frequencies
            word_counts = {}
            for text in texts:
                words = text.lower().split()
                for word in words:
                    if len(word) > 2 and word not in stopwords and word.isalpha():
                        word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get top words
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            keywords = [word for word, count in top_words]
            
            # Create German label
            german_label = self._create_contextual_german_label(keywords, texts)
            
            cluster_meta[cluster_id] = {
                'label': german_label,
                'keywords': keywords,
                'size': len(texts)
            }
        
        return cluster_meta
    
    def _get_german_stopwords(self) -> set:
        """Get comprehensive German stopwords list."""
        stopwords = set()
        
        # Try to get NLTK German stopwords
        if self.capabilities['nltk']:
            try:
                import nltk
                from nltk.corpus import stopwords as nltk_stopwords
                nltk.download('stopwords', quiet=True)
                stopwords.update(nltk_stopwords.words('german'))
            except Exception as e:
                logger.warning(f"Could not load NLTK stopwords: {e}")
        
        # Comprehensive German stopwords
        german_stopwords = {
            # Articles and pronouns
            'der', 'die', 'das', 'und', 'ist', 'ich', 'nicht', 'sie', 'es', 'ein', 'eine', 'einen', 'einer',
            'dem', 'den', 'des', 'ihm', 'ihn', 'ihr', 'ihre', 'ihren', 'ihrer', 'ihres', 'mein', 'meine',
            
            # Prepositions and conjunctions
            'mit', 'zu', 'auf', 'für', 'von', 'in', 'an', 'bei', 'nach', 'über', 'unter', 'aus', 'durch',
            'dass', 'wenn', 'aber', 'oder', 'auch', 'noch', 'nur', 'schon', 'sehr', 'wie', 'was', 'wo',
            
            # Verbs
            'haben', 'sein', 'werden', 'können', 'müssen', 'sollen', 'wollen', 'dürfen', 'hat', 'hatte',
            'wurde', 'wurden', 'würde', 'würden', 'kann', 'konnte', 'könnte', 'sollte', 'wollte', 'war',
            'waren', 'wird', 'sind', 'bin', 'bist', 'wurde', 'worden',
            
            # Common English words (for mixed content)
            'this', 'the', 'and', 'is', 'was', 'are', 'were', 'been', 'have', 'has', 'had', 'will', 'would',
            'can', 'could', 'should', 'would', 'may', 'might', 'must', 'shall', 'do', 'does', 'did'
        }
        stopwords.update(german_stopwords)
        
        return stopwords
    
    def _create_contextual_german_label(self, terms: List[str], texts: List[str]) -> str:
        """
        Create meaningful German cluster labels based on context analysis.
        
        Args:
            terms (List[str]): Top terms from TF-IDF analysis
            texts (List[str]): Sample texts from the cluster
            
        Returns:
            str: Contextual German label
        """
        # Business domain term mapping for German labels
        term_mapping = {
            # Customer service
            'support': 'Support', 'customer': 'Kunden', 'service': 'Service', 'help': 'Hilfe',
            'agent': 'Mitarbeiter', 'staff': 'Personal', 'team': 'Team',
            
            # Technical issues
            'login': 'Anmeldung', 'timeout': 'Timeout', 'timeouts': 'Timeouts', 'error': 'Fehler',
            'bug': 'Fehler', 'issue': 'Problem', 'problem': 'Problem', 'crash': 'Absturz',
            
            # Communication
            'callback': 'Rückruf', 'promised': 'Versprochen', 'call': 'Anruf', 'email': 'E-Mail',
            'chat': 'Chat', 'response': 'Antwort', 'reply': 'Antwort',
            
            # Features and functionality
            'sync': 'Synchronisation', 'calendar': 'Kalender', 'helpful': 'Hilfreich',
            'dashboard': 'Dashboard', 'layout': 'Layout', 'new': 'Neu', 'update': 'Update',
            'export': 'Export', 'excel': 'Excel', 'csv': 'CSV', 'pdf': 'PDF',
            
            # Performance and quality
            'performance': 'Performance', 'speed': 'Geschwindigkeit', 'slow': 'Langsam',
            'fast': 'Schnell', 'quick': 'Schnell', 'loading': 'Laden',
            
            # Business processes
            'billing': 'Abrechnung', 'invoice': 'Rechnung', 'payment': 'Zahlung',
            'shipping': 'Versand', 'delivery': 'Lieferung', 'order': 'Bestellung',
            
            # User interface
            'app': 'App', 'mobile': 'Mobil', 'ui': 'Oberfläche', 'interface': 'Oberfläche',
            'navigation': 'Navigation', 'menu': 'Menü', 'button': 'Button'
        }
        
        # Analyze sample texts for context
        sample_text = ' '.join(texts[:3]).lower()
        
        # Create contextual labels based on dominant themes
        if any(word in sample_text for word in ['support', 'customer', 'service', 'hilfe']):
            if any(word in sample_text for word in ['callback', 'rückruf', 'call']):
                return 'Support Rückruf'
            elif any(word in sample_text for word in ['convenient', 'praktisch', 'good', 'gut']):
                return 'Support Qualität'
            else:
                return 'Kundensupport'
                
        elif any(word in sample_text for word in ['login', 'anmeld', 'timeout']):
            return 'Anmelde-Probleme'
            
        elif any(word in sample_text for word in ['sync', 'synchron', 'calendar', 'kalender']):
            return 'Synchronisation'
            
        elif any(word in sample_text for word in ['dashboard', 'layout', 'ui', 'interface']):
            return 'Benutzeroberfläche'
            
        elif any(word in sample_text for word in ['shipping', 'delivery', 'lieferung', 'versand']):
            return 'Versand & Lieferung'
            
        elif any(word in sample_text for word in ['pdf', 'download', 'export']):
            return 'Datei-Download'
            
        elif any(word in sample_text for word in ['billing', 'invoice', 'rechnung', 'payment']):
            return 'Abrechnung & Zahlung'
            
        elif any(word in sample_text for word in ['performance', 'speed', 'slow', 'fast']):
            return 'System-Performance'
            
        else:
            # Fallback: use translated terms
            german_terms = []
            for term in terms[:2]:
                if term in term_mapping:
                    german_terms.append(term_mapping[term])
                elif len(term) > 3:
                    german_terms.append(term.title())
            
            if german_terms:
                return ' & '.join(german_terms)
            else:
                return f"Thema {abs(hash(' '.join(terms))) % 100}"
    
    def analyze_sentiment(self, texts: List[str]) -> Tuple[List[float], List[str]]:
        """
        Perform sentiment analysis on German text.
        
        Args:
            texts (List[str]): List of text messages
            
        Returns:
            Tuple[List[float], List[str]]: Sentiment scores and categories
        """
        if self.capabilities['vader']:
            logger.info("Performing VADER sentiment analysis")
            return self._vader_sentiment_analysis(texts)
        else:
            logger.info("Performing rule-based German sentiment analysis")
            return self._german_sentiment_analysis(texts)
    
    def _vader_sentiment_analysis(self, texts: List[str]) -> Tuple[List[float], List[str]]:
        """Advanced sentiment analysis using VADER."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        
        scores = []
        categories = []
        
        for text in texts:
            # VADER works reasonably well with German text
            score = analyzer.polarity_scores(text or "")['compound']
            scores.append(score)
            
            # Classify sentiment with appropriate thresholds
            if score <= -0.05:
                category = "negative"
            elif score >= 0.05:
                category = "positive"
            else:
                category = "neutral"
            categories.append(category)
        
        return scores, categories
    
    def _german_sentiment_analysis(self, texts: List[str]) -> Tuple[List[float], List[str]]:
        """Rule-based sentiment analysis optimized for German text."""
        # Comprehensive German sentiment lexicons
        positive_words = {
            'gut', 'super', 'toll', 'prima', 'perfekt', 'ausgezeichnet', 'fantastisch',
            'wunderbar', 'großartig', 'hervorragend', 'zufrieden', 'freundlich',
            'schnell', 'pünktlich', 'hilfsbereit', 'kompetent', 'professionell',
            'excellent', 'great', 'good', 'amazing', 'wonderful', 'perfect',
            'helpful', 'friendly', 'fast', 'quick', 'satisfied', 'happy'
        }
        
        negative_words = {
            'schlecht', 'schrecklich', 'furchtbar', 'katastrophe', 'unzufrieden',
            'langsam', 'unpünktlich', 'unfreundlich', 'inkompetent', 'problem',
            'fehler', 'mangel', 'beschwerde', 'ärgerlich', 'enttäuscht', 'frustriert',
            'bad', 'terrible', 'awful', 'horrible', 'slow', 'poor', 'disappointed',
            'frustrated', 'angry', 'upset', 'unsatisfied', 'complaint'
        }
        
        scores = []
        categories = []
        
        for text in texts:
            words = text.lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            # Calculate normalized sentiment score
            total_words = len(words)
            if total_words == 0:
                score = 0.0
                category = 'neutral'
            else:
                score = (pos_count - neg_count) / total_words
                
                # Classify with appropriate thresholds
                if score > 0.02:
                    category = 'positive'
                elif score < -0.02:
                    category = 'negative'
                else:
                    category = 'neutral'
            
            scores.append(score)
            categories.append(category)
        
        return scores, categories
    
    def create_visualizations(self, df: pd.DataFrame):
        """
        Create comprehensive German visualizations for the analysis results.
        
        Args:
            df (pd.DataFrame): Analysis results dataframe
        """
        if not self.capabilities['matplotlib']:
            logger.warning("Matplotlib not available - skipping visualizations")
            logger.info("Install matplotlib and pyparsing for visualizations: pip install matplotlib pyparsing")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Try to import seaborn for better styling
            if self.capabilities['seaborn']:
                import seaborn as sns
                sns.set_style("whitegrid")
            
            self.output_dir.mkdir(exist_ok=True)
            
            # Set German locale for better formatting
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            
            # Create comprehensive dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Semantische Kundenfeedback-Analyse Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Top 10 German Clusters
            top_clusters = df[df['cluster_id'] != -1]['cluster_label'].value_counts().head(10)
            if len(top_clusters) > 0:
                y_pos = range(len(top_clusters))
                axes[0,0].barh(y_pos, top_clusters.values, color='skyblue', edgecolor='navy', alpha=0.7)
                axes[0,0].set_yticks(y_pos)
                axes[0,0].set_yticklabels([self._truncate_label(label, 25) for label in top_clusters.index])
                axes[0,0].set_title('Top 10 Semantische Cluster', fontweight='bold')
                axes[0,0].set_xlabel('Anzahl Nachrichten')
                axes[0,0].grid(axis='x', alpha=0.3)
            
            # 2. Sentiment Distribution with German labels
            sentiment_counts = df['sentiment_category'].value_counts()
            german_sentiment = {'positive': 'Positiv', 'neutral': 'Neutral', 'negative': 'Negativ'}
            colors = {'positive': '#2E8B57', 'neutral': '#708090', 'negative': '#DC143C'}
            
            if len(sentiment_counts) > 0:
                labels = [german_sentiment.get(cat, cat) for cat in sentiment_counts.index]
                colors_list = [colors.get(cat, 'gray') for cat in sentiment_counts.index]
                
                wedges, texts, autotexts = axes[0,1].pie(
                    sentiment_counts.values, 
                    labels=labels, 
                    autopct='%1.1f%%',
                    colors=colors_list,
                    startangle=90
                )
                axes[0,1].set_title('Sentiment Verteilung', fontweight='bold')
            
            # 3. Monthly Sentiment Trend
            monthly_sentiment = df.groupby('year_month')['sentiment_score'].mean()
            if len(monthly_sentiment) > 0:
                axes[1,0].plot(monthly_sentiment.index, monthly_sentiment.values, 
                              marker='o', linewidth=2, markersize=6, color='#4169E1')
                axes[1,0].set_title('Durchschnittliches Sentiment über Zeit', fontweight='bold')
                axes[1,0].set_ylabel('Sentiment Score')
                axes[1,0].tick_params(axis='x', rotation=45)
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Neutral')
                axes[1,0].legend()
            
            # 4. Cluster Sentiment Heatmap
            if self.capabilities['seaborn'] and len(df) > 0:
                cluster_sentiment = df.groupby(['cluster_label', 'sentiment_category']).size().unstack(fill_value=0)
                top_clusters_for_heatmap = df['cluster_label'].value_counts().head(8).index
                
                if len(cluster_sentiment) > 0:
                    cluster_sentiment_top = cluster_sentiment.loc[
                        cluster_sentiment.index.intersection(top_clusters_for_heatmap)
                    ]
                    
                    if len(cluster_sentiment_top) > 0:
                        sns.heatmap(cluster_sentiment_top, annot=True, fmt='d', 
                                   cmap='RdYlGn', ax=axes[1,1], cbar_kws={'label': 'Anzahl'})
                        axes[1,1].set_title('Sentiment nach Top Clustern', fontweight='bold')
                        axes[1,1].set_ylabel('Cluster')
                        axes[1,1].set_xlabel('Sentiment Kategorie')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'semantic_clustering_dashboard.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info("German visualization dashboard created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _truncate_label(self, label: str, max_length: int) -> str:
        """Truncate long labels for better visualization."""
        return label[:max_length] + '...' if len(label) > max_length else label
    
    def save_results(self, df: pd.DataFrame, summary: pd.DataFrame):
        """
        Save analysis results in multiple formats.
        
        Args:
            df (pd.DataFrame): Complete analysis results
            summary (pd.DataFrame): Cluster summary statistics
        """
        self.output_dir.mkdir(exist_ok=True)
        
        try:
            # Always save CSV (universal compatibility)
            df.to_csv(self.output_dir / 'feedback_analysis_results.csv', 
                     index=False, encoding='utf-8')
            summary.to_csv(self.output_dir / 'cluster_summary.csv', 
                          index=False, encoding='utf-8')
            
            # Save Excel if available (better for business users)
            if self.capabilities['openpyxl']:
                with pd.ExcelWriter(self.output_dir / 'feedback_analysis_complete.xlsx', 
                                   engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Vollständige Analyse', index=False)
                    summary.to_excel(writer, sheet_name='Cluster Zusammenfassung', index=False)
                
                logger.info("Results saved in CSV and Excel formats")
            else:
                logger.info("Results saved in CSV format")
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete German feedback analysis pipeline.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Analysis results and summary
            
        Raises:
            Exception: If analysis fails at any stage
        """
        logger.info("=" * 60)
        logger.info("STARTING GERMAN CUSTOMER FEEDBACK ANALYSIS")
        logger.info("=" * 60)
        
        try:
            # 1. Data Loading and Cleaning
            logger.info("Step 1: Loading and cleaning data...")
            self.df = self.load_data()
            self.df = self.clean_data(self.df)
            
            if len(self.df) == 0:
                raise ValueError("No valid data remaining after cleaning")
            
            texts = self.df['nachricht'].tolist()
            
            # 2. Text Embedding Creation
            logger.info("Step 2: Creating text embeddings...")
            X = self.create_embeddings(texts)
            
            # 3. Dimensionality Reduction
            logger.info("Step 3: Applying dimensionality reduction...")
            X_reduced = self.reduce_dimensions(X)
            
            # 4. Semantic Clustering
            logger.info("Step 4: Performing semantic clustering...")
            labels = self.cluster_texts(X_reduced)
            self.df['cluster_id'] = labels
            
            # 5. German Cluster Labeling
            logger.info("Step 5: Generating German cluster labels...")
            self.cluster_meta = self.generate_cluster_labels(texts, labels)
            
            # Add cluster labels to dataframe
            self.df['cluster_label'] = self.df['cluster_id'].map(
                lambda x: self.cluster_meta.get(x, {}).get('label', 'Unbekannt')
            )
            
            # 6. Sentiment Analysis
            logger.info("Step 6: Analyzing sentiment...")
            sentiment_scores, sentiment_categories = self.analyze_sentiment(texts)
            self.df['sentiment_score'] = sentiment_scores
            self.df['sentiment_category'] = sentiment_categories
            
            # 7. Summary Creation
            logger.info("Step 7: Creating analysis summary...")
            summary = self._create_summary()
            
            # 8. Visualization Generation
            logger.info("Step 8: Creating visualizations...")
            self.create_visualizations(self.df)
            
            # 9. Results Saving
            logger.info("Step 9: Saving results...")
            self.save_results(self.df, summary)
            
            # 10. Final Report
            self._print_final_report(summary)
            
            logger.info("Analysis completed successfully!")
            return self.df, summary
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _create_summary(self) -> pd.DataFrame:
        """Create comprehensive summary statistics."""
        try:
            summary = self.df.groupby('cluster_id').agg({
                'nachricht': 'count',
                'sentiment_score': ['mean', 'std'],
                'year_month': lambda x: x.mode().iloc[0] if len(x) > 0 else None
            })
            
            # Flatten column names
            summary.columns = ['message_count', 'avg_sentiment', 'sentiment_std', 'peak_month']
            
            # Add cluster metadata
            summary['cluster_label'] = [
                self.cluster_meta.get(idx, {}).get('label', 'Unbekannt') 
                for idx in summary.index
            ]
            summary['keywords'] = [
                ', '.join(self.cluster_meta.get(idx, {}).get('keywords', [])[:5])
                for idx in summary.index
            ]
            
            # Calculate additional metrics
            summary['sentiment_std'] = summary['sentiment_std'].fillna(0)
            summary['percentage'] = (summary['message_count'] / len(self.df) * 100).round(1)
            
            # Sort by message count
            summary = summary.sort_values('message_count', ascending=False)
            
            return summary.reset_index()
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return pd.DataFrame()
    
    def _print_final_report(self, summary: pd.DataFrame):
        """Print comprehensive German analysis report."""
        n_messages = len(self.df)
        n_clusters = len([idx for idx in summary['cluster_id'] if idx != -1])
        avg_sentiment = self.df['sentiment_score'].mean()
        
        print("\n" + "=" * 70)
        print("DEUTSCHE KUNDENFEEDBACK-ANALYSE ABGESCHLOSSEN")
        print("=" * 70)
        
        # Basic statistics
        print(f"📊 ANALYSE STATISTIKEN:")
        print(f"   • Nachrichten analysiert: {n_messages:,}")
        print(f"   • Semantische Cluster gefunden: {n_clusters}")
        print(f"   • Durchschnittliches Sentiment: {avg_sentiment:.3f}")
        
        # Technology stack used
        advanced_features = [k for k, v in self.capabilities.items() if v]
        print(f"   • Verwendete Technologien: {', '.join(advanced_features)}")
        
        # Sentiment distribution
        sentiment_dist = self.df['sentiment_category'].value_counts()
        print(f"\n💭 SENTIMENT VERTEILUNG:")
        for category, count in sentiment_dist.items():
            percentage = (count / n_messages) * 100
            emoji = "😊" if category == "positive" else "😐" if category == "neutral" else "😞"
            print(f"   {emoji} {category.title()}: {count:,} ({percentage:.1f}%)")
        
        # Top themes
        print(f"\n🎯 TOP 5 THEMEN:")
        top_clusters = summary[summary['cluster_id'] != -1].head(5)
        for i, row in enumerate(top_clusters.itertuples(), 1):
            sentiment_indicator = "📈" if row.avg_sentiment > 0.1 else "📉" if row.avg_sentiment < -0.1 else "➡️"
            print(f"   {i}. {row.cluster_label}")
            print(f"      {sentiment_indicator} {row.message_count} Nachrichten ({row.percentage}%) | "
                  f"Sentiment: {row.avg_sentiment:.3f}")
        
        # Output files
        print(f"\n📁 ERGEBNISSE GESPEICHERT:")
        print(f"   • Vollständige Analyse: output/feedback_analysis_results.csv")
        print(f"   • Cluster Zusammenfassung: output/cluster_summary.csv")
        if self.capabilities['openpyxl']:
            print(f"   • Excel Datei: output/feedback_analysis_complete.xlsx")
        if self.capabilities['matplotlib']:
            print(f"   • Visualisierung: output/semantic_clustering_dashboard.png")
        
        print(f"\n✅ Analyse erfolgreich abgeschlossen!")
        print("=" * 70)


def main():
    """
    Main function to run the German feedback analysis.
    
    This function serves as the entry point for the analysis system.
    It can be customized for different input files or configurations.
    """
    try:
        # Initialize analyzer with default settings
        analyzer = GermanFeedbackAnalyzer()
        
        # Run complete analysis
        results_df, summary_df = analyzer.run_analysis()
        
        return results_df, summary_df
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()