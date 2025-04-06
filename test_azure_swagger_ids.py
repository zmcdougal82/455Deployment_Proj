#!/usr/bin/env python3
import urllib.request
import json
import sys

def test_azure_with_swagger_ids():
    """
    Test the Azure ML endpoint using the IDs from the Swagger documentation example
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Use the exact IDs from the Swagger documentation example
    swagger_ids = [
        {"personId": 1908339160857512799, "contentId": 3460026829794173084, "eventType": 1},
        {"personId": -445337111692715325, "contentId": -7820640624231356730, "eventType": 1},
        {"personId": 4254153380739593270, "contentId": -1492913151930215984, "eventType": 1}
    ]
    
    # Try each ID individually
    for i, id_data in enumerate(swagger_ids):
        print(f"\n=== Testing with Swagger ID {i+1} ===")
        
        # Try as user recommendation
        print(f"\nTrying as user recommendation (personId: {id_data['personId']}):")
        data = {
            "Inputs": {
                "input1": [
                    {
                        "personId": id_data['personId'],
                        "contentId": 0,
                        "eventType": id_data['eventType']
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
        
        # Make the request
        try:
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
        
        # Try as item recommendation
        print(f"\nTrying as item recommendation (contentId: {id_data['contentId']}):")
        data = {
            "Inputs": {
                "input1": [
                    {
                        "personId": 0,
                        "contentId": id_data['contentId'],
                        "eventType": id_data['eventType']
                    }
                ]
            },
            "GlobalParameters": {
                "requestType": "item_recommendations",
                "numRecommendations": 5
            }
        }
        
        print("Request data:")
        print(json.dumps(data, indent=2))
        
        # Make the request
        try:
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
    Test the Azure ML endpoint with a batch of IDs from the Swagger documentation example
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Use the exact batch from the Swagger documentation example
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
        "GlobalParameters": {
            "requestType": "user_recommendations",
            "numRecommendations": 5
        }
    }
    
    print("\n=== Testing with Batch of IDs ===")
    print("Request data:")
    print(json.dumps(data, indent=2))
    
    # Make the request
    try:
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

def test_azure_with_different_params():
    """
    Test the Azure ML endpoint with different GlobalParameters
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Try different GlobalParameters
    params_list = [
        {},  # Empty GlobalParameters
        {"requestType": "user_recommendations"},  # Just requestType
        {"numRecommendations": 5},  # Just numRecommendations
        {"requestType": "item_recommendations", "numRecommendations": 5},  # Item recommendations
        {"requestType": "user_recommendations", "numRecommendations": 10},  # More recommendations
        {"requestType": "user_recommendations", "numRecommendations": 5, "includeMetadata": True},  # Additional parameter
        {"requestType": "user_recommendations", "numRecommendations": 5, "minScore": 0.5}  # Additional parameter
    ]
    
    # Use the first ID from the Swagger documentation example
    for i, params in enumerate(params_list):
        print(f"\n=== Testing with GlobalParameters {i+1} ===")
        
        data = {
            "Inputs": {
                "input1": [
                    {
                        "personId": 1908339160857512799,
                        "contentId": 0,
                        "eventType": 1
                    }
                ]
            },
            "GlobalParameters": params
        }
        
        print("Request data:")
        print(json.dumps(data, indent=2))
        
        # Make the request
        try:
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
    # Test with IDs from the Swagger documentation example
    test_azure_with_swagger_ids()
    
    # Test with a batch of IDs from the Swagger documentation example
    test_azure_with_batch()
    
    # Test with different GlobalParameters
    test_azure_with_different_params()

if __name__ == "__main__":
    main()
