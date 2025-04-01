#!/usr/bin/env python3
import pickle
import sys
import os

# Add the models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import the model classes
from simple_collaborative_filtering import SimpleCollaborativeFilteringModel
from simple_content_filtering import ContentFilteringModel

def main():
    # Load the collaborative model
    print("Loading collaborative model...")
    with open('models/collaborative_model.sav', 'rb') as f:
        collab_model = pickle.load(f)
    
    # Get some valid user IDs
    valid_user_ids = list(collab_model.user_mapping.keys())[:5]
    print(f"Valid user IDs: {valid_user_ids}")
    
    # Get some valid item IDs
    valid_item_ids = list(collab_model.item_mapping.keys())[:5]
    print(f"Valid item IDs: {valid_item_ids}")
    
    # Load the content model
    print("\nLoading content model...")
    with open('models/content_model.sav', 'rb') as f:
        content_model = pickle.load(f)
    
    # Get some valid content IDs
    valid_content_ids = content_model.article_ids[:5]
    print(f"Valid content IDs: {valid_content_ids}")

if __name__ == "__main__":
    main()
