#!/usr/bin/env python3
import urllib.request
import json
import sys

def test_azure_with_string_event():
    """
    Test the Azure ML endpoint using a string value for eventType
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Use a string value for eventType as in the JavaScript implementation
    data = {
        "Inputs": {
            "input1": [
                {
                    "personId": 6756039155228175109,
                    "contentId": 0,
                    "eventType": "click"  # Using string value as in the JavaScript implementation
                }
            ]
        },
        "GlobalParameters": {
            "requestType": "user_recommendations",
            "numRecommendations": 5
        }
    }
    
    print("Request data:")
    print(json.dumps(data, indent=2))
    
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
        
        # Print the raw response
        print("\nResponse:")
        print(json.dumps(result_json, indent=2))
        
        # Check if we got any data
        if result_json and result_json != {"Results": {}}:
            print("\nSuccess! Got a non-empty response")
        else:
            print("\nGot an empty response")
    except Exception as e:
        print(f"\nError: {e}")
        if isinstance(e, urllib.error.HTTPError):
            print(f"HTTP Error: {e.code}")
            print(f"Response: {e.read().decode('utf-8')}")

def test_azure_with_different_event_values():
    """
    Test the Azure ML endpoint with different string values for eventType
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Try different string values for eventType
    event_types = ["click", "view", "like", "purchase", "rating"]
    
    for event_type in event_types:
        print(f"\n=== Testing with eventType = '{event_type}' ===")
        
        data = {
            "Inputs": {
                "input1": [
                    {
                        "personId": 6756039155228175109,
                        "contentId": 0,
                        "eventType": event_type
                    }
                ]
            },
            "GlobalParameters": {
                "requestType": "user_recommendations",
                "numRecommendations": 5
            }
        }
        
        print("Request data:")
        print(json.dumps(data, indent=2))
        
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
            
            # Print the raw response
            print("\nResponse:")
            print(json.dumps(result_json, indent=2))
            
            # Check if we got any data
            if result_json and result_json != {"Results": {}}:
                print("\nSuccess! Got a non-empty response")
            else:
                print("\nGot an empty response")
        except Exception as e:
            print(f"\nError: {e}")
            if isinstance(e, urllib.error.HTTPError):
                print(f"HTTP Error: {e.code}")
                print(f"Response: {e.read().decode('utf-8')}")

def main():
    # Test with a string value for eventType
    test_azure_with_string_event()
    
    # Test with different string values for eventType
    test_azure_with_different_event_values()

if __name__ == "__main__":
    main()
