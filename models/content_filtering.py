#!/usr/bin/env python3
import sys
import json
import pickle
import os

# Import the model class
from simple_content_filtering import ContentFilteringModel

def load_model(model_path):
    """Load the content filtering model from a .sav file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None

def get_recommendations(model, item_id, top_n=5):
    """
    Get recommendations using the content filtering model
    
    Parameters:
    - model: The loaded content filtering model
    - item_id: The ID of the item to get similar items for
    - top_n: Number of recommendations to return
    
    Returns:
    - List of recommended item IDs with scores
    """
    # Convert ID to string to ensure consistent handling
    item_id = str(item_id)
    
    # Get similar items based on content
    # Redirect model's print statements to stderr
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    recommendations = model.get_similar_items(item_id, top_n)
    sys.stdout = original_stdout
    
    return recommendations

def main():
    """Main function to handle command line arguments and return recommendations as JSON"""
    if len(sys.argv) < 2:
        print("Usage: python content_filtering.py <item_id>", file=sys.stderr)
        sys.exit(1)
    
    # Get command line arguments
    item_id = sys.argv[1]
    
    # Path to the model file
    model_path = os.path.join(os.path.dirname(__file__), 'content_filtering.sav')
    
    # Load the model
    model = load_model(model_path)
    if not model:
        print(json.dumps({"error": "Failed to load model"}))
        sys.exit(1)
    
    # Get recommendations
    recommendations = get_recommendations(model, item_id)
    
    # Return recommendations as JSON
    print(json.dumps(recommendations))

if __name__ == "__main__":
    main()
