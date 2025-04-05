#!/usr/bin/env python3
import pickle
import sys
import json
import os

def main():
    try:
        # Load the collaborative model
        print("Loading collaborative model...")
        with open(os.path.join('models', 'collaborative_model.sav'), 'rb') as f:
            collab_model = pickle.load(f)
        
        # Get some valid user IDs
        user_ids = list(collab_model.user_mapping.keys())
        print(f"Found {len(user_ids)} user IDs")
        print(f"Sample user IDs: {user_ids[:5]}")
        
        # Get some valid item IDs
        item_ids = list(collab_model.item_mapping.keys())
        print(f"Found {len(item_ids)} item IDs")
        print(f"Sample item IDs: {item_ids[:5]}")
        
        # Load the content model
        print("\nLoading content model...")
        with open(os.path.join('models', 'content_filtering.sav'), 'rb') as f:
            content_model = pickle.load(f)
        
        # Get some valid content IDs
        content_ids = content_model.article_ids
        print(f"Found {len(content_ids)} content IDs")
        print(f"Sample content IDs: {content_ids[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
