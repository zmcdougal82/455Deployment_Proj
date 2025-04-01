#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from collections import defaultdict

# Define paths
DATA_PATH = 'users_interactions.csv'
MODEL_PATH = 'models/collaborative_model.sav'

class CollaborativeFilteringModel:
    def __init__(self):
        self.model = None
        self.user_mapping = {}  # Maps original user IDs to internal IDs
        self.item_mapping = {}  # Maps original item IDs to internal IDs
        self.reverse_user_mapping = {}  # Maps internal IDs back to original user IDs
        self.reverse_item_mapping = {}  # Maps internal IDs back to original item IDs
        self.user_items = defaultdict(list)  # Items rated by each user
        self.item_users = defaultdict(list)  # Users who rated each item
        
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
        
        # Create a new DataFrame with user_id, item_id, and rating
        ratings_data = []
        for _, row in interactions_df.iterrows():
            user_id = self.user_mapping[row['personId']]
            item_id = self.item_mapping[row['contentId']]
            event_type = row['eventType']
            rating = event_type_mapping.get(event_type, 1.0)  # Default to 1.0 if event type not found
            
            ratings_data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            })
            
            # Store user-item and item-user relationships
            self.user_items[user_id].append(item_id)
            self.item_users[item_id].append(user_id)
        
        ratings_df = pd.DataFrame(ratings_data)
        
        # Create a Surprise dataset
        reader = Reader(rating_scale=(1, 2))
        dataset = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
        
        # Build the full trainset
        trainset = dataset.build_full_trainset()
        
        print("Training model...")
        # Use SVD algorithm for matrix factorization with reduced complexity for faster training
        self.model = SVD(n_factors=20, n_epochs=5, lr_all=0.01, reg_all=0.02)
        self.model.fit(trainset)
        
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
        all_items = set(self.reverse_item_mapping.keys())
        items_to_predict = list(all_items - user_items)
        
        # If no items to predict, return empty list
        if not items_to_predict:
            return []
        
        # Predict ratings for all items the user has not interacted with
        predictions = []
        for item_id in items_to_predict:
            predicted_rating = self.model.predict(internal_user_id, item_id).est
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
        
        # Get all items except the input item
        all_items = set(self.reverse_item_mapping.keys())
        all_items.remove(internal_item_id)
        
        # Calculate similarity between the input item and all other items
        similarities = []
        for other_item_id in all_items:
            # Get users who rated both items
            item_users = set(self.item_users[internal_item_id])
            other_item_users = set(self.item_users[other_item_id])
            common_users = item_users.intersection(other_item_users)
            
            if not common_users:
                continue
            
            # Calculate similarity based on model factors
            item_factors = self.model.qi[internal_item_id]
            other_item_factors = self.model.qi[other_item_id]
            
            # Cosine similarity
            similarity = np.dot(item_factors, other_item_factors) / (
                np.linalg.norm(item_factors) * np.linalg.norm(other_item_factors)
            )
            
            similarities.append((other_item_id, similarity))
        
        # Sort similarities in descending order and take top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_n]
        
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

def main():
    """
    Main function to build and save the collaborative filtering model
    """
    print("Loading data...")
    try:
        # Load only necessary columns to save memory
        interactions_df = pd.read_csv(DATA_PATH, usecols=['personId', 'contentId', 'eventType'])
        
        # Remove rows with missing values
        interactions_df = interactions_df.dropna(subset=['personId', 'contentId', 'eventType'])
        
        # Convert IDs to strings to ensure consistent handling
        interactions_df['personId'] = interactions_df['personId'].astype(str)
        interactions_df['contentId'] = interactions_df['contentId'].astype(str)
        
        # Sample data for faster processing
        print("Sampling data for faster processing...")
        interactions_df = interactions_df.sample(frac=0.05, random_state=42)
        
        print(f"Loaded {len(interactions_df)} interactions")
        
        # Create and train the model
        model = CollaborativeFilteringModel()
        model.fit(interactions_df)
        
        # Save the model
        print(f"Saving model to {MODEL_PATH}...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error building model: {e}")
        
if __name__ == "__main__":
    main()
