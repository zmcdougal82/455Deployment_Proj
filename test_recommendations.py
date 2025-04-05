#!/usr/bin/env python3
import pickle
import sys
import os
import json

# Import the model classes
sys.path.append('models')
from simple_collaborative_filtering import SimpleCollaborativeFilteringModel
from simple_content_filtering import ContentFilteringModel

def main():
    try:
        # Test collaborative model
        print("Testing collaborative model...")
        collab_model_path = os.path.join('models', 'collaborative_model.sav')
        
        if not os.path.exists(collab_model_path):
            print(f"Error: Collaborative model file not found at {collab_model_path}")
            return
        
        with open(collab_model_path, 'rb') as f:
            collab_model = pickle.load(f)
        
        print(f"Collaborative model loaded successfully")
        print(f"Model type: {type(collab_model)}")
        
        # Test user recommendations
        user_id = "-88452987812994280181264196770339959068"
        print(f"\nTesting user recommendations for user ID: {user_id}")
        
        if hasattr(collab_model, 'user_mapping'):
            if user_id in collab_model.user_mapping:
                print(f"User ID found in model's user_mapping")
                user_recs = collab_model.get_user_recommendations(user_id, top_n=5)
                print(f"User recommendations: {json.dumps(user_recs, indent=2)}")
            else:
                print(f"User ID not found in model's user_mapping")
                print(f"Available user IDs: {list(collab_model.user_mapping.keys())[:5]}")
        else:
            print(f"Model does not have user_mapping attribute")
        
        # Test item recommendations
        item_id = "-645130951826674502"
        print(f"\nTesting item recommendations for item ID: {item_id}")
        
        if hasattr(collab_model, 'item_mapping'):
            if item_id in collab_model.item_mapping:
                print(f"Item ID found in model's item_mapping")
                item_recs = collab_model.get_similar_items(item_id, top_n=5)
                print(f"Item recommendations: {json.dumps(item_recs, indent=2)}")
            else:
                print(f"Item ID not found in model's item_mapping")
                print(f"Available item IDs: {list(collab_model.item_mapping.keys())[:5]}")
        else:
            print(f"Model does not have item_mapping attribute")
        
        # Test content model
        print("\nTesting content model...")
        content_model_path = os.path.join('models', 'content_filtering.sav')
        
        if not os.path.exists(content_model_path):
            print(f"Error: Content model file not found at {content_model_path}")
            return
        
        with open(content_model_path, 'rb') as f:
            content_model = pickle.load(f)
        
        print(f"Content model loaded successfully")
        print(f"Model type: {type(content_model)}")
        
        # Test content recommendations
        content_id = "-645130951826674502"
        print(f"\nTesting content recommendations for content ID: {content_id}")
        
        if hasattr(content_model, 'article_ids'):
            if content_id in content_model.article_ids:
                print(f"Content ID found in model's article_ids")
                content_recs = content_model.get_similar_items(content_id, top_n=5)
                print(f"Content recommendations: {json.dumps(content_recs, indent=2)}")
            else:
                print(f"Content ID not found in model's article_ids")
                print(f"Available content IDs: {content_model.article_ids[:5]}")
        else:
            print(f"Model does not have article_ids attribute")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
