#!/usr/bin/env python3
import urllib.request
import json
import sys
import random

def test_azure_with_random_ids():
    """
    Test the Azure ML endpoint with random IDs
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'Ydnjdd8tr1wwAtpCJG7bftYOQYPXxO2j'
    
    # Generate 5 random IDs
    random_ids = [random.randint(1000000000, 9999999999) for _ in range(5)]
    
    for i, random_id in enumerate(random_ids):
        print(f"\n=== Testing with Random ID {i+1}: {random_id} ===")
        
        # Try as user recommendation
        data = {
            "Inputs": {
                "input1": [
                    {
                        "personId": random_id,
                        "contentId": 0,
                        "eventType": 1
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

def test_azure_with_different_request_format():
    """
    Test the Azure ML endpoint with a different request format
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'Ydnjdd8tr1wwAtpCJG7bftYOQYPXxO2j'
    
    # Try a completely different request format
    request_formats = [
        # Format 1: Minimal request with just the required fields
        {
            "Inputs": {
                "input1": [
                    {
                        "personId": 1234567890,
                        "contentId": 0,
                        "eventType": 1
                    }
                ]
            }
        },
        
        # Format 2: Request with WebServiceInput instead of Inputs
        {
            "WebServiceInput": {
                "input1": [
                    {
                        "personId": 1234567890,
                        "contentId": 0,
                        "eventType": 1
                    }
                ]
            },
            "GlobalParameters": {
                "requestType": "user_recommendations",
                "numRecommendations": 5
            }
        },
        
        # Format 3: Request with data field instead of Inputs
        {
            "data": [
                {
                    "personId": 1234567890,
                    "contentId": 0,
                    "eventType": 1
                }
            ],
            "GlobalParameters": {
                "requestType": "user_recommendations",
                "numRecommendations": 5
            }
        },
        
        # Format 4: Request with different GlobalParameters
        {
            "Inputs": {
                "input1": [
                    {
                        "personId": 1234567890,
                        "contentId": 0,
                        "eventType": 1
                    }
                ]
            },
            "GlobalParameters": {
                "method": "user_recommendations",
                "count": 5
            }
        },
        
        # Format 5: Request with multiple records
        {
            "Inputs": {
                "input1": [
                    {
                        "personId": 1234567890,
                        "contentId": 0,
                        "eventType": 1
                    },
                    {
                        "personId": 9876543210,
                        "contentId": 0,
                        "eventType": 1
                    },
                    {
                        "personId": 5555555555,
                        "contentId": 0,
                        "eventType": 1
                    }
                ]
            },
            "GlobalParameters": {
                "requestType": "user_recommendations",
                "numRecommendations": 5
            }
        }
    ]
    
    for i, request_data in enumerate(request_formats):
        print(f"\n=== Testing with Request Format {i+1} ===")
        print("Request data:")
        print(json.dumps(request_data, indent=2))
        
        try:
            # Convert data to JSON string and then to bytes
            body = str.encode(json.dumps(request_data))
            
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

def test_azure_with_different_event_types():
    """
    Test the Azure ML endpoint with different event types
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'Ydnjdd8tr1wwAtpCJG7bftYOQYPXxO2j'
    
    # Try different event types
    event_types = [0, 1, 2, 3, 4, 5, 10, 100]
    
    for event_type in event_types:
        print(f"\n=== Testing with eventType = {event_type} ===")
        
        data = {
            "Inputs": {
                "input1": [
                    {
                        "personId": 1234567890,
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
    print("=== Testing with Random IDs ===")
    test_azure_with_random_ids()
    
    print("\n=== Testing with Different Request Formats ===")
    test_azure_with_different_request_format()
    
    print("\n=== Testing with Different Event Types ===")
    test_azure_with_different_event_types()

if __name__ == "__main__":
    main()
