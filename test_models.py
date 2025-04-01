#!/usr/bin/env python3
import pickle
import sys
import os
import sys

# Add the models directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import the model classes
from simple_collaborative_filtering import SimpleCollaborativeFilteringModel
from simple_content_filtering import ContentFilteringModel

def test_load_model(model_path):
    print(f"Attempting to load model from {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model: {type(model)}")
        print(f"Model attributes: {dir(model)}")
        
        # Test a method
        if hasattr(model, 'get_similar_items'):
            print("Testing get_similar_items method...")
            try:
                # Try with a dummy item ID
                result = model.get_similar_items("1", top_n=2)
                print(f"Method returned: {result}")
            except Exception as e:
                print(f"Error calling method: {e}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    # Test collaborative model
    print("Testing collaborative model...")
    collab_success = test_load_model('models/collaborative_model.sav')
    
    # Test content model
    print("\nTesting content model...")
    content_success = test_load_model('models/content_model.sav')
    
    # Summary
    print("\nSummary:")
    print(f"Collaborative model: {'SUCCESS' if collab_success else 'FAILED'}")
    print(f"Content model: {'SUCCESS' if content_success else 'FAILED'}")
