#!/usr/bin/env python3
import pickle
import sys
import json
import os

# Import the model classes
sys.path.append('models')
from simple_collaborative_filtering import SimpleCollaborativeFilteringModel
from simple_content_filtering import ContentFilteringModel

def inspect_collaborative_model():
    """Inspect the collaborative filtering model"""
    print("\n=== Collaborative Filtering Model ===")
    try:
        with open(os.path.join('models', 'collaborative_model.sav'), 'rb') as f:
            collab_model = pickle.load(f)
        
        print(f"Model type: {type(collab_model)}")
        
        # Inspect attributes
        print("\nModel attributes:")
        for attr in dir(collab_model):
            if not attr.startswith('__') and not callable(getattr(collab_model, attr)):
                attr_value = getattr(collab_model, attr)
                if isinstance(attr_value, (dict, list, set)):
                    print(f"  {attr}: {type(attr_value)} with {len(attr_value)} items")
                else:
                    print(f"  {attr}: {type(attr_value)}")
        
        # Get some sample user IDs
        if hasattr(collab_model, 'user_mapping'):
            user_ids = list(collab_model.user_mapping.keys())
            print(f"\nSample user IDs ({min(5, len(user_ids))} of {len(user_ids)}):")
            for i, user_id in enumerate(user_ids[:5]):
                print(f"  {i+1}. {user_id}")
        
        # Get some sample item IDs
        if hasattr(collab_model, 'item_mapping'):
            item_ids = list(collab_model.item_mapping.keys())
            print(f"\nSample item IDs ({min(5, len(item_ids))} of {len(item_ids)}):")
            for i, item_id in enumerate(item_ids[:5]):
                print(f"  {i+1}. {item_id}")
        
        # Check if there's any interaction data
        if hasattr(collab_model, 'user_item_matrix'):
            matrix = collab_model.user_item_matrix
            print(f"\nUser-item matrix shape: {matrix.shape}")
            
            # Count non-zero entries (interactions)
            non_zero = (matrix > 0).sum()
            print(f"Number of interactions: {non_zero}")
            
            # Sample some interactions
            if non_zero > 0:
                print("\nSample interactions:")
                count = 0
                for i in range(min(100, matrix.shape[0])):
                    for j in range(min(100, matrix.shape[1])):
                        if matrix[i, j] > 0:
                            user_id = collab_model.reverse_user_mapping.get(i, f"User_{i}")
                            item_id = collab_model.reverse_item_mapping.get(j, f"Item_{j}")
                            print(f"  User {user_id} interacted with item {item_id} (value: {matrix[i, j]})")
                            count += 1
                            if count >= 5:
                                break
                    if count >= 5:
                        break
        
        return collab_model
    except Exception as e:
        print(f"Error inspecting collaborative model: {e}")
        return None

def inspect_content_model():
    """Inspect the content filtering model"""
    print("\n=== Content Filtering Model ===")
    try:
        with open(os.path.join('models', 'content_filtering.sav'), 'rb') as f:
            content_model = pickle.load(f)
        
        print(f"Model type: {type(content_model)}")
        
        # Inspect attributes
        print("\nModel attributes:")
        for attr in dir(content_model):
            if not attr.startswith('__') and not callable(getattr(content_model, attr)):
                attr_value = getattr(content_model, attr)
                if isinstance(attr_value, (dict, list, set)):
                    print(f"  {attr}: {type(attr_value)} with {len(attr_value)} items")
                else:
                    print(f"  {attr}: {type(attr_value)}")
        
        # Get some sample content IDs
        if hasattr(content_model, 'article_ids'):
            content_ids = content_model.article_ids
            print(f"\nSample content IDs ({min(5, len(content_ids))} of {len(content_ids)}):")
            for i, content_id in enumerate(content_ids[:5]):
                print(f"  {i+1}. {content_id}")
        
        return content_model
    except Exception as e:
        print(f"Error inspecting content model: {e}")
        return None

def try_azure_with_model_data(collab_model, content_model):
    """Try to use data from the models to make a request to the Azure ML endpoint"""
    print("\n=== Trying Azure ML with Model Data ===")
    
    # Import the Azure ML function
    sys.path.append('.')
    from test_azure_ml import get_azure_ml_recommendations
    
    # Try with a user ID from the collaborative model
    if collab_model and hasattr(collab_model, 'user_mapping') and collab_model.user_mapping:
        user_id = list(collab_model.user_mapping.keys())[0]
        print(f"\nTrying Azure ML with user ID: {user_id}")
        user_recommendations = get_azure_ml_recommendations(user_id, 'user', 5)
        print(f"User recommendations: {json.dumps(user_recommendations, indent=2)}")
    
    # Try with an item ID from the collaborative model
    if collab_model and hasattr(collab_model, 'item_mapping') and collab_model.item_mapping:
        item_id = list(collab_model.item_mapping.keys())[0]
        print(f"\nTrying Azure ML with item ID: {item_id}")
        item_recommendations = get_azure_ml_recommendations(item_id, 'item', 5)
        print(f"Item recommendations: {json.dumps(item_recommendations, indent=2)}")
    
    # Try with a content ID from the content model
    if content_model and hasattr(content_model, 'article_ids') and content_model.article_ids:
        content_id = content_model.article_ids[0]
        print(f"\nTrying Azure ML with content ID: {content_id}")
        content_recommendations = get_azure_ml_recommendations(content_id, 'item', 5)
        print(f"Content recommendations: {json.dumps(content_recommendations, indent=2)}")

def main():
    # Inspect the collaborative filtering model
    collab_model = inspect_collaborative_model()
    
    # Inspect the content filtering model
    content_model = inspect_content_model()
    
    # Try to use data from the models to make a request to the Azure ML endpoint
    try_azure_with_model_data(collab_model, content_model)

if __name__ == "__main__":
    main()
