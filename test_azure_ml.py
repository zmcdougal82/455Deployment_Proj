#!/usr/bin/env python3
import urllib.request
import json
import sys
import re
import os
import pickle

def get_azure_ml_recommendations(id_value, id_type='user', num_recommendations=5, event_type=0):
    """
    Get recommendations from the Azure ML endpoint
    
    Parameters:
    - id_value: The ID of the user or item to get recommendations for
    - id_type: 'user' or 'item' - whether to get recommendations for a user or similar items
    - num_recommendations: Number of recommendations to return
    - event_type: The event type value to use in the request (default: 0)
    
    Returns:
    - List of recommended item IDs with scores
    """
    # Azure ML endpoint configuration
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")
    
    # Convert ID to integer if possible, otherwise use 0
    try:
        id_int = int(id_value)
    except ValueError:
        id_int = 0
    
    # Prepare data for Azure ML endpoint based on the expected schema
    data = {
        "Inputs": {
            "input1": [
                {
                    "personId": id_int if id_type == 'user' else 0,
                    "contentId": id_int if id_type == 'item' else 0,
                    "eventType": event_type  # Using the provided event_type value
                }
            ]
        },
        "GlobalParameters": {
            "requestType": "user_recommendations" if id_type == 'user' else "item_recommendations",
            "numRecommendations": num_recommendations
        }
    }
    
    # Convert data to JSON string and then to bytes
    body = str.encode(json.dumps(data))
    
    # Set up headers
    headers = {
        'Content-Type': 'application/json', 
        'Accept': 'application/json', 
        'Authorization': 'Bearer ' + api_key
    }
    
    # Create request
    req = urllib.request.Request(url, body, headers)
    
    try:
        # Send request and get response
        response = urllib.request.urlopen(req)
        result = response.read()
        
        # Parse JSON response
        result_json = json.loads(result)
        
        # Print the raw response for debugging
        print("Raw Azure ML response:", file=sys.stderr)
        print(json.dumps(result_json, indent=2), file=sys.stderr)
        
        # Process the Azure ML response based on its structure
        recommendations = []
        
        # Handle different response formats
        if 'Results' in result_json and 'output1' in result_json['Results']:
            # Standard Azure ML format with Results.output1
            output_data = result_json['Results']['output1']
            
            if isinstance(output_data, list):
                recommendations = [
                    {
                        'contentId': str(item.get('contentId') or item.get('Recommended_contentId') or f"item_{i}"),
                        'score': float(item.get('score') or item.get('Score') or 0.5),
                        'reason': f"Azure ML recommendation for {id_type} {id_value}"
                    }
                    for i, item in enumerate(output_data)
                ]
            elif isinstance(output_data, dict):
                # Handle case where output is a single object
                recommendations = [{
                    'contentId': str(output_data.get('contentId') or output_data.get('Recommended_contentId') or 'item_0'),
                    'score': float(output_data.get('score') or output_data.get('Score') or 0.5),
                    'reason': f"Azure ML recommendation for {id_type} {id_value}"
                }]
        elif 'value' in result_json and isinstance(result_json['value'], list):
            # Another common Azure ML format
            recommendations = [
                {
                    'contentId': str(item.get('contentId') or item.get('Recommended_contentId') or f"item_{i}"),
                    'score': float(item.get('score') or item.get('Score') or 0.5),
                    'reason': f"Azure ML recommendation for {id_type} {id_value}"
                }
                for i, item in enumerate(result_json['value'])
            ]
        elif isinstance(result_json, list):
            # If response is already an array of recommendations
            recommendations = [
                {
                    'contentId': str(item.get('contentId') or item.get('Recommended_contentId') or f"item_{i}"),
                    'score': float(item.get('score') or item.get('Score') or 0.5),
                    'reason': f"Azure ML recommendation for {id_type} {id_value}"
                }
                for i, item in enumerate(result_json)
            ]
        else:
            # Fallback: try to extract any useful data from the response
            print("Using fallback response handling for Azure ML", file=sys.stderr)
            
            # Try to find arrays in the response that might contain recommendations
            candidate_arrays = []
            
            def find_arrays(obj, path=''):
                if not obj or not isinstance(obj, dict):
                    return
                
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, list) and len(value) > 0:
                        candidate_arrays.append({'path': current_path, 'data': value})
                    elif isinstance(value, dict):
                        find_arrays(value, current_path)
            
            find_arrays(result_json)
            
            if candidate_arrays:
                # Use the first array found that has objects with potential recommendation data
                best_candidate = None
                for candidate in candidate_arrays:
                    if any(isinstance(item, dict) and (
                        'contentId' in item or 
                        'itemId' in item or 
                        'Recommended_contentId' in item or 
                        'score' in item or 
                        'Score' in item
                    ) for item in candidate['data']):
                        best_candidate = candidate
                        break
                
                if not best_candidate:
                    best_candidate = candidate_arrays[0]
                
                recommendations = []
                for i, item in enumerate(best_candidate['data']):
                    if isinstance(item, dict):
                        recommendations.append({
                            'contentId': str(item.get('contentId') or item.get('itemId') or item.get('Recommended_contentId') or f"item_{i}"),
                            'score': float(item.get('score') or item.get('Score') or 0.5),
                            'reason': f"Azure ML recommendation for {id_type} {id_value}"
                        })
                    else:
                        recommendations.append({
                            'contentId': f"item_{i}",
                            'score': float(item) if isinstance(item, (int, float)) else 0.5,
                            'reason': f"Azure ML recommendation for {id_type} {id_value}"
                        })
            else:
                # If no arrays found, create dummy recommendations
                print("No suitable arrays found in Azure ML response, creating dummy recommendations", file=sys.stderr)
                recommendations = [
                    {
                        'contentId': f"item_{i}",
                        'score': 0.5 - (i * 0.05),
                        'reason': f"Fallback recommendation for {id_type} {id_value} (Azure ML response format not recognized)"
                    }
                    for i in range(5)
                ]
        
        # Limit to specified number of recommendations and sort by score
        recommendations = sorted(recommendations[:num_recommendations], key=lambda x: x['score'], reverse=True)
        
        return recommendations
        
    except urllib.error.HTTPError as error:
        print(f"The request failed with status code: {error.code}", file=sys.stderr)
        print(error.info(), file=sys.stderr)
        print(error.read().decode("utf8", 'ignore'), file=sys.stderr)
        return {"error": f"HTTP Error: {error.code}", "message": error.read().decode("utf8", 'ignore')}
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        return {"error": "Exception", "message": str(e)}

def get_schema_info():
    """
    Attempt to get schema information from the Azure ML endpoint
    by making a request with an intentionally incorrect format
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Create a request with all required fields but with an incorrect type for eventType
    data = {
        "Inputs": {
            "input1": [
                {
                    "personId": 123,
                    "contentId": 456,
                    "eventType": "incorrect_type"  # Intentionally use a string to trigger a type error
                }
            ]
        }
    }
    
    body = str.encode(json.dumps(data))
    
    headers = {
        'Content-Type': 'application/json', 
        'Accept': 'application/json', 
        'Authorization': 'Bearer ' + api_key
    }
    
    req = urllib.request.Request(url, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        print("Unexpected success:", file=sys.stderr)
        print(result.decode("utf8"), file=sys.stderr)
    except urllib.error.HTTPError as error:
        print("Schema error information:", file=sys.stderr)
        error_content = error.read().decode("utf8", 'ignore')
        print(error_content, file=sys.stderr)
        
        # Try to extract schema information from the error message
        try:
            error_json = json.loads(error_content)
            if 'error' in error_json and 'message' in error_json['error']:
                error_message = error_json['error']['message']
                
                # Look for schema information in the error message
                schema_start = error_message.find('Schema:')
                if schema_start != -1:
                    schema_info = error_message[schema_start:]
                    schema_end = schema_info.find('\n')
                    if schema_end != -1:
                        schema_info = schema_info[:schema_end]
                    
                    print("\nExtracted Schema Information:", file=sys.stderr)
                    print(schema_info, file=sys.stderr)
                    
                    # Try to parse the schema information
                    try:
                        # Extract column names and types
                        column_info = []
                        column_matches = re.findall(r"'name': '([^']+)'[^}]+'type': '([^']+)'", schema_info)
                        
                        if column_matches:
                            print("\nColumn Information:", file=sys.stderr)
                            for name, type_info in column_matches:
                                print(f"Column: {name}, Type: {type_info}", file=sys.stderr)
                                column_info.append((name, type_info))
                            
                            return column_info
                    except Exception as e:
                        print(f"Error parsing schema: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing error content: {e}", file=sys.stderr)
    
    return None

def try_different_combinations(id_value, id_type='user', num_recommendations=5):
    """
    Try different combinations of personId, contentId, and eventType values
    """
    print(f"\nTrying different combinations for {id_type} {id_value}:")
    
    # Get some sample IDs from the models
    sys.path.append('models')
    try:
        from simple_collaborative_filtering import SimpleCollaborativeFilteringModel
        from simple_content_filtering import ContentFilteringModel
        
        # Load the collaborative model to get user and item IDs
        with open(os.path.join('models', 'collaborative_model.sav'), 'rb') as f:
            collab_model = pickle.load(f)
        
        # Get some sample user IDs
        user_ids = list(collab_model.user_mapping.keys())[:3]
        
        # Get some sample item IDs
        item_ids = list(collab_model.item_mapping.keys())[:3]
        
        # Load the content model to get content IDs
        with open(os.path.join('models', 'content_filtering.sav'), 'rb') as f:
            content_model = pickle.load(f)
        
        # Get some sample content IDs
        content_ids = content_model.article_ids[:3]
        
        print(f"Sample user IDs: {user_ids}")
        print(f"Sample item IDs: {item_ids}")
        print(f"Sample content IDs: {content_ids}")
    except Exception as e:
        print(f"Error loading models: {e}")
        # Use some default IDs
        user_ids = ['6756039155228175109', '-3933783680725097100', '8195788452563155020']
        item_ids = ['5940374562401786524', '3170775058142440102', '-5799839529845993396']
        content_ids = ['5036201777135800491', '2822049545552366036', '8181531374206668183']
    
    # Try different event_type values
    event_types = [0, 1, 2]
    
    # Try different combinations
    combinations = []
    
    # If id_type is 'user', try different contentId values
    if id_type == 'user':
        for event_type in event_types:
            for content_id in item_ids + content_ids:
                combinations.append({
                    'personId': id_value,
                    'contentId': content_id,
                    'eventType': event_type
                })
    # If id_type is 'item', try different personId values
    else:
        for event_type in event_types:
            for user_id in user_ids:
                combinations.append({
                    'personId': user_id,
                    'contentId': id_value,
                    'eventType': event_type
                })
    
    # Try each combination
    for i, combo in enumerate(combinations):
        print(f"\nTrying combination {i+1}/{len(combinations)}:")
        print(f"  personId: {combo['personId']}")
        print(f"  contentId: {combo['contentId']}")
        print(f"  eventType: {combo['eventType']}")
        
        # Make the request
        url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
        api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
        
        # Convert IDs to integers if possible
        try:
            person_id = int(combo['personId'])
        except ValueError:
            person_id = 0
        
        try:
            content_id = int(combo['contentId'])
        except ValueError:
            content_id = 0
        
        # Prepare data
        data = {
            "Inputs": {
                "input1": [
                    {
                        "personId": person_id,
                        "contentId": content_id,
                        "eventType": combo['eventType']
                    }
                ]
            },
            "GlobalParameters": {
                "requestType": "user_recommendations" if id_type == 'user' else "item_recommendations",
                "numRecommendations": num_recommendations
            }
        }
        
        # Convert data to JSON string and then to bytes
        body = str.encode(json.dumps(data))
        
        # Set up headers
        headers = {
            'Content-Type': 'application/json', 
            'Accept': 'application/json', 
            'Authorization': 'Bearer ' + api_key
        }
        
        # Create request
        req = urllib.request.Request(url, body, headers)
        
        try:
            # Send request and get response
            response = urllib.request.urlopen(req)
            result = response.read()
            
            # Parse JSON response
            result_json = json.loads(result)
            
            # Print the raw response for debugging
            print("Raw Azure ML response:")
            print(json.dumps(result_json, indent=2))
            
            # Check if we got actual recommendations
            if 'Results' in result_json and 'output1' in result_json['Results'] and result_json['Results']['output1']:
                print(f"Success! Found working combination")
                return result_json
        except Exception as e:
            print(f"Error: {e}")
    
    print("None of the combinations worked")
    return None

def main():
    """Main function to handle command line arguments and return recommendations as JSON"""
    if len(sys.argv) < 2:
        print("Usage: python test_azure_ml.py <id> [user|item] [num_recommendations]", file=sys.stderr)
        print("       python test_azure_ml.py schema", file=sys.stderr)
        print("       python test_azure_ml.py try_events <id> [user|item]", file=sys.stderr)
        sys.exit(1)
    
    # Check if the user wants to get schema information
    if sys.argv[1].lower() == 'schema':
        get_schema_info()
        sys.exit(0)
    
    # Check if the user wants to try different combinations
    if sys.argv[1].lower() == 'try_events' and len(sys.argv) >= 3:
        id_value = sys.argv[2]
        id_type = sys.argv[3] if len(sys.argv) > 3 else 'user'
        try_different_combinations(id_value, id_type)
        sys.exit(0)
    
    # Get command line arguments
    id_value = sys.argv[1]
    id_type = sys.argv[2] if len(sys.argv) > 2 else 'user'
    num_recommendations = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    # Get recommendations
    recommendations = get_azure_ml_recommendations(id_value, id_type, num_recommendations)
    
    # Print recommendations as JSON
    print(json.dumps(recommendations, indent=2))

if __name__ == "__main__":
    main()
