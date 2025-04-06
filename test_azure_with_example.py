#!/usr/bin/env python3
import urllib.request
import json
import sys

def test_azure_with_example():
    """
    Test the Azure ML endpoint using the example from the Swagger documentation
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Use the exact example from the Swagger documentation
    data = {
        "Inputs": {
            "input1": [
                {
                    "personId": 1908339160857512799,
                    "contentId": 3460026829794173084,
                    "eventType": 1
                },
                {
                    "personId": -445337111692715325,
                    "contentId": -7820640624231356730,
                    "eventType": 1
                },
                {
                    "personId": 4254153380739593270,
                    "contentId": -1492913151930215984,
                    "eventType": 1
                }
            ]
        },
        "GlobalParameters": {}
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

def test_azure_with_batch():
    """
    Test the Azure ML endpoint with a batch of user IDs from our models
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Use user IDs from our models
    user_ids = [
        6756039155228175109,
        -3933783680725097100,
        8195788452563155020,
        -1616903969205976623,
        6409254426985102122
    ]
    
    # Create a batch request with multiple user IDs
    input_records = []
    for user_id in user_ids:
        input_records.append({
            "personId": user_id,
            "contentId": 0,  # Use 0 for contentId when getting user recommendations
            "eventType": 1   # Use 1 for eventType as shown in the example
        })
    
    data = {
        "Inputs": {
            "input1": input_records
        },
        "GlobalParameters": {
            "requestType": "user_recommendations",
            "numRecommendations": 5
        }
    }
    
    print("\n=== Testing with Batch of User IDs ===")
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
    # Test with the example from the Swagger documentation
    test_azure_with_example()
    
    # Test with a batch of user IDs from our models
    test_azure_with_batch()

if __name__ == "__main__":
    main()
