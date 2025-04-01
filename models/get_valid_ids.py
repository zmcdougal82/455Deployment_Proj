#!/usr/bin/env python3
import pickle
import sys
import json
import os

# Add the models directory to the Python path
sys.path.append(os.path.dirname(__file__))

# Import the model classes
from simple_collaborative_filtering import SimpleCollaborativeFilteringModel
from simple_content_filtering import ContentFilteringModel

def main():
    try:
        # Load the collaborative model
        with open(os.path.join(os.path.dirname(__file__), 'collaborative_model.sav'), 'rb') as f:
            collab_model = pickle.load(f)
        
        # Get some valid user IDs
        valid_user_ids = [str(id) for id in list(collab_model.user_mapping.keys())[:10]]
        
        # Get some valid item IDs
        valid_item_ids = [str(id) for id in list(collab_model.item_mapping.keys())[:10]]
        
        # Load the content model
        with open(os.path.join(os.path.dirname(__file__), 'content_model.sav'), 'rb') as f:
            content_model = pickle.load(f)
        
        # Get some valid content IDs
        valid_content_ids = [str(id) for id in content_model.article_ids[:10]]
        
        # Create a dictionary of valid IDs
        valid_ids = {
            'users': valid_user_ids,
            'items': valid_item_ids,
            'content': valid_content_ids
        }
        
        # Return the valid IDs as JSON
        print(json.dumps(valid_ids))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
