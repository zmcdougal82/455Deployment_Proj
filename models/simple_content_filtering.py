#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class ContentFilteringModel:
    def __init__(self):
        self.vectorizer = None
        self.content_vectors = None
        self.article_ids = None
        self.article_titles = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters, converting to lowercase,
        removing stopwords, and stemming
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        
        # Tokenize, remove stopwords, and stem
        tokens = [self.stemmer.stem(word) for word in text.split() if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def fit(self, articles_df):
        """
        Train the content-based filtering model
        
        Parameters:
        - articles_df: DataFrame with columns [contentId, title, text]
        """
        print("Preprocessing article content...")
        
        # Store article IDs and titles for later use
        self.article_ids = articles_df['contentId'].astype(str).tolist()
        self.article_titles = articles_df['title'].tolist()
        
        # Combine title and text for better content representation
        # Title is repeated to give it more weight
        articles_df['combined_content'] = articles_df['title'].fillna('') + ' ' + \
                                         articles_df['title'].fillna('') + ' ' + \
                                         articles_df['text'].fillna('')
        
        # Preprocess the combined content
        articles_df['processed_content'] = articles_df['combined_content'].apply(self.preprocess_text)
        
        print("Vectorizing article content...")
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.content_vectors = self.vectorizer.fit_transform(articles_df['processed_content'])
        
        print("Model training complete!")
        print(f"Vectorized {self.content_vectors.shape[0]} articles with {self.content_vectors.shape[1]} features")
        
    def get_similar_items(self, item_id, top_n=5):
        """
        Get similar items to a given item based on content similarity
        
        Parameters:
        - item_id: The ID of the item to get similar items for
        - top_n: Number of similar items to return
        
        Returns:
        - List of similar item IDs with scores
        """
        # Convert item_id to string for consistent handling
        item_id = str(item_id)
        
        # Check if the item exists in our dataset
        if item_id not in self.article_ids:
            print(f"Item {item_id} not found in the dataset")
            return []
        
        # Get the index of the item
        item_index = self.article_ids.index(item_id)
        
        # Get the content vector for the item
        item_vector = self.content_vectors[item_index]
        
        # Calculate similarity with all other items
        similarity_scores = cosine_similarity(item_vector, self.content_vectors).flatten()
        
        # Get indices of top similar items (excluding the item itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        
        # Create recommendations list
        recommendations = []
        for idx in similar_indices:
            # Get content-based reason
            reason = self._get_content_reason(item_index, idx)
            
            recommendations.append({
                'contentId': self.article_ids[idx],
                'score': float(similarity_scores[idx]),
                'reason': reason
            })
        
        return recommendations
    
    def _get_content_reason(self, item_index, similar_index):
        """
        Generate a content-based reason for the recommendation
        """
        item_title = self.article_titles[item_index] if item_index < len(self.article_titles) else "this article"
        similar_title = self.article_titles[similar_index] if similar_index < len(self.article_titles) else "this article"
        
        # List of possible content-based reasons
        reasons = [
            f"Similar topics to '{item_title}'",
            f"Similar writing style to '{item_title}'",
            f"Similar keywords to '{item_title}'",
            f"Content similar to '{item_title}'"
        ]
        
        # For simplicity, choose a random reason
        # In a real system, you would analyze the specific features that make the items similar
        import random
        return random.choice(reasons)

if __name__ == "__main__":
    # Define paths
    DATA_PATH = 'shared_articles.csv'
    MODEL_PATH = 'models/content_model.sav'
    
    # Load only necessary columns to save memory
    print("Loading data...")
    articles_df = pd.read_csv(DATA_PATH, usecols=['contentId', 'title', 'text', 'lang'])
    
    # Filter for English articles only
    articles_df = articles_df[articles_df['lang'] == 'en']
    
    # Remove rows with missing title and text
    articles_df = articles_df.dropna(subset=['title', 'text'])
    
    # Convert IDs to strings to ensure consistent handling
    articles_df['contentId'] = articles_df['contentId'].astype(str)
    
    # Sample data for faster processing
    print("Sampling data for faster processing...")
    articles_df = articles_df.sample(frac=0.2, random_state=42)
    print(f"Sampled {len(articles_df)} articles")
    
    # Create and train the model
    print("Creating and training the model...")
    model = ContentFilteringModel()
    model.fit(articles_df)
    
    # Test the model with a random article
    random_article = articles_df['contentId'].sample(1).iloc[0]
    article_title = articles_df[articles_df['contentId'] == random_article]['title'].iloc[0]
    print(f"\nTesting model with random article: {random_article}")
    print(f"Article title: {article_title}")
    
    # Get similar articles
    similar_articles = model.get_similar_items(random_article, top_n=5)
    print("\nSimilar articles:")
    for i, article in enumerate(similar_articles):
        article_id = article['contentId']
        title = articles_df[articles_df['contentId'] == article_id]['title'].iloc[0] if article_id in articles_df['contentId'].values else "Unknown"
        print(f"{i+1}. {title}")
        print(f"   Content ID: {article_id}, Score: {article['score']:.4f}")
        print(f"   Reason: {article['reason']}")
    
    # Save the model
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")
