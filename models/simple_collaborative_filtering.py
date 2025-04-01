#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict

class SimpleCollaborativeFilteringModel:
    def __init__(self):
        self.user_mapping = {}  # Maps original user IDs to internal IDs
        self.item_mapping = {}  # Maps original item IDs to internal IDs
        self.reverse_user_mapping = {}  # Maps internal IDs back to original user IDs
        self.reverse_item_mapping = {}  # Maps internal IDs back to original item IDs
        self.user_items = defaultdict(list)  # Items rated by each user
        self.item_users = defaultdict(list)  # Users who rated each item
        self.user_item_matrix = None  # User-item interaction matrix
        self.item_similarity_matrix = None  # Item-item similarity matrix
        
    def fit(self, interactions_df):
        """
        Train the collaborative filtering model
        
        Parameters:
        - interactions_df: DataFrame with columns [personId, contentId, eventType]
        """
        print("Preprocessing data...")
        
        # Create user and item mappings
        unique_users = interactions_df['personId'].unique()
        unique_items = interactions_df['contentId'].unique()
        
        self.user_mapping = {user: i for i, user in enumerate(unique_users)}
        self.item_mapping = {item: i for i, item in enumerate(unique_items)}
        
        self.reverse_user_mapping = {i: user for user, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}
        
        # Convert eventType to numerical ratings
        # VIEW = 1, FOLLOW = 2, etc. (can be customized based on your data)
        event_type_mapping = {
            'VIEW': 1.0,
            'FOLLOW': 2.0
        }
        
        # Create user-item matrix
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        # Fill the user-item matrix
        for _, row in interactions_df.iterrows():
            user_id = self.user_mapping[row['personId']]
            item_id = self.item_mapping[row['contentId']]
            event_type = row['eventType']
            rating = event_type_mapping.get(event_type, 1.0)  # Default to 1.0 if event type not found
            
            self.user_item_matrix[user_id, item_id] = rating
            
            # Store user-item and item-user relationships
            self.user_items[user_id].append(item_id)
            self.item_users[item_id].append(user_id)
        
        print("Computing item-item similarity matrix...")
        # Compute item-item similarity matrix using cosine similarity
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-9
        item_norms = np.sqrt(np.sum(self.user_item_matrix**2, axis=0)) + epsilon
        self.item_similarity_matrix = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(i, n_items):
                # Compute cosine similarity
                dot_product = np.sum(self.user_item_matrix[:, i] * self.user_item_matrix[:, j])
                similarity = dot_product / (item_norms[i] * item_norms[j])
                self.item_similarity_matrix[i, j] = similarity
                self.item_similarity_matrix[j, i] = similarity  # Symmetric matrix
        
        print("Model training complete!")
        
    def get_user_recommendations(self, user_id, top_n=5):
        """
        Get recommendations for a user
        
        Parameters:
        - user_id: The ID of the user to get recommendations for
        - top_n: Number of recommendations to return
        
        Returns:
        - List of recommended item IDs with scores
        """
        if user_id not in self.user_mapping:
            # If user not in training data, return empty list
            return []
        
        internal_user_id = self.user_mapping[user_id]
        
        # Get items the user has not interacted with
        user_items = set(self.user_items[internal_user_id])
        all_items = set(range(len(self.reverse_item_mapping)))
        items_to_predict = list(all_items - user_items)
        
        # If no items to predict, return empty list
        if not items_to_predict:
            return []
        
        # Predict ratings for all items the user has not interacted with
        predictions = []
        for item_id in items_to_predict:
            # Get items the user has interacted with
            user_interacted_items = self.user_items[internal_user_id]
            
            if not user_interacted_items:
                continue
            
            # Compute weighted average of similarities
            weighted_sum = 0
            similarity_sum = 0
            for interacted_item in user_interacted_items:
                similarity = self.item_similarity_matrix[item_id, interacted_item]
                rating = self.user_item_matrix[internal_user_id, interacted_item]
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)
            
            # Avoid division by zero
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
            else:
                predicted_rating = 0
            
            predictions.append((item_id, predicted_rating))
        
        # Sort predictions by rating in descending order and take top_n
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]
        
        # Convert internal item IDs back to original IDs
        recommendations = []
        for item_id, score in top_predictions:
            original_item_id = self.reverse_item_mapping[item_id]
            recommendations.append({
                'contentId': str(original_item_id),
                'score': float(score),
                'reason': f'Based on your reading history (User {user_id})'
            })
        
        return recommendations
    
    def get_similar_items(self, item_id, top_n=5):
        """
        Get similar items to a given item
        
        Parameters:
        - item_id: The ID of the item to get similar items for
        - top_n: Number of similar items to return
        
        Returns:
        - List of similar item IDs with scores
        """
        if item_id not in self.item_mapping:
            # If item not in training data, return empty list
            return []
        
        internal_item_id = self.item_mapping[item_id]
        
        # Get similarity scores for all items
        similarity_scores = self.item_similarity_matrix[internal_item_id]
        
        # Create a list of (item_id, similarity) tuples
        item_similarities = [(i, similarity_scores[i]) for i in range(len(similarity_scores)) if i != internal_item_id]
        
        # Sort by similarity in descending order and take top_n
        item_similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = item_similarities[:top_n]
        
        # Convert internal item IDs back to original IDs
        similar_items = []
        for other_item_id, score in top_similarities:
            original_item_id = self.reverse_item_mapping[other_item_id]
            similar_items.append({
                'contentId': str(original_item_id),
                'score': float(score),
                'reason': f'Similar to article {item_id}'
            })
        
        return similar_items

if __name__ == "__main__":
    # Define paths
    DATA_PATH = 'users_interactions.csv'
    MODEL_PATH = 'models/collaborative_model.sav'
    
    # Load only necessary columns to save memory
    print("Loading data...")
    interactions_df = pd.read_csv(DATA_PATH, usecols=['personId', 'contentId', 'eventType'])
    
    # Remove rows with missing values
    interactions_df = interactions_df.dropna(subset=['personId', 'contentId', 'eventType'])
    
    # Convert IDs to strings to ensure consistent handling
    interactions_df['personId'] = interactions_df['personId'].astype(str)
    interactions_df['contentId'] = interactions_df['contentId'].astype(str)
    
    # Sample data for faster processing
    print("Sampling data for faster processing...")
    interactions_df = interactions_df.sample(frac=0.05, random_state=42)
    print(f"Sampled {len(interactions_df)} interactions")
    
    # Create and train the model
    print("Creating and training the model...")
    model = SimpleCollaborativeFilteringModel()
    model.fit(interactions_df)
    
    # Test the model with a random user
    random_user = interactions_df['personId'].sample(1).iloc[0]
    print(f"\nTesting model with random user: {random_user}")
    user_recommendations = model.get_user_recommendations(random_user, top_n=5)
    print("\nUser recommendations:")
    for i, rec in enumerate(user_recommendations):
        print(f"{i+1}. Content ID: {rec['contentId']}, Score: {rec['score']:.4f}, Reason: {rec['reason']}")
    
    # Test the model with a random item
    random_item = interactions_df['contentId'].sample(1).iloc[0]
    print(f"\nTesting model with random item: {random_item}")
    similar_items = model.get_similar_items(random_item, top_n=5)
    print("\nSimilar items:")
    for i, item in enumerate(similar_items):
        print(f"{i+1}. Content ID: {item['contentId']}, Score: {item['score']:.4f}, Reason: {item['reason']}")
    
    # Save the model
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")
