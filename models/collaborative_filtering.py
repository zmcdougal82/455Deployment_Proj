#!/usr/bin/env python3
import sys
import json
import pickle
import os

# Import the model class
from simple_collaborative_filtering import SimpleCollaborativeFilteringModel

def load_model(model_path):
    """Load the collaborative filtering model from a .sav file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None

def get_recommendations(model, id_value, id_type='item', top_n=5):
    """
    Get recommendations using the collaborative filtering model
    
    Parameters:
    - model: The loaded collaborative filtering model
    - id_value: The ID of the user or item to get recommendations for
    - id_type: 'user' or 'item' - whether to get recommendations for a user or similar items
    - top_n: Number of recommendations to return
    
    Returns:
    - List of recommended item IDs with scores
    """
    try:
        # Convert ID to string to ensure consistent handling
        id_value = str(id_value)
        
        # Redirect model's print statements to stderr
        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        
        # Use the appropriate model method based on id_type
        if id_type == 'user':
            # Get recommendations for a user
            recommendations = model.get_user_recommendations(id_value, top_n)
        else:
            # Get similar items
            recommendations = model.get_similar_items(id_value, top_n)
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # If no recommendations were found, return fallback recommendations
        if not recommendations:
            print(f"No recommendations found for {id_type} {id_value}", file=sys.stderr)
            return [
                {"contentId": f"fallback-{i}", "score": 0.5, "reason": f"Fallback recommendation for {id_type} {id_value}"} 
                for i in range(top_n)
            ]
        
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {e}", file=sys.stderr)
        # Even if there's an error, return some fallback recommendations
        # so the frontend doesn't break
        return [
            {"contentId": f"fallback-{i}", "score": 0.5, "reason": "Fallback recommendation"} 
            for i in range(top_n)
        ]

def main():
    """Main function to handle command line arguments and return recommendations as JSON"""
    if len(sys.argv) < 2:
        print("Usage: python collaborative_filtering.py <id> [user|item]", file=sys.stderr)
        sys.exit(1)
    
    # Get command line arguments
    id_value = sys.argv[1]
    id_type = sys.argv[2] if len(sys.argv) > 2 else 'item'
    
    # Path to the model file
    model_path = os.path.join(os.path.dirname(__file__), 'collaborative_model.sav')
    
    # Load the model
    model = load_model(model_path)
    if not model:
        print(json.dumps({"error": "Failed to load model"}))
        sys.exit(1)
    
    # Get recommendations
    recommendations = get_recommendations(model, id_value, id_type)
    
    # Return recommendations as JSON
    print(json.dumps(recommendations))

if __name__ == "__main__":
    main()
