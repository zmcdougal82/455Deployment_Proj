#!/usr/bin/env python3
import urllib.request
import json
import sys

def test_azure_endpoint():
    """
    Test the Azure ML endpoint with various request formats to see if we can get any response
    """
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    api_key = 'UJDerMxrslnaVUrmW8XIoZm6xSktLrWA'
    
    # Try different request formats
    request_formats = [
        # Format 1: Empty request
        {},
        
        # Format 2: Just Inputs with empty array
        {
            "Inputs": {
                "input1": []
            }
        },
        
        # Format 3: Just GlobalParameters
        {
            "GlobalParameters": {
                "requestType": "user_recommendations",
                "numRecommendations": 5
            }
        },
        
        # Format 4: Different input structure
        {
            "data": [
                {
                    "personId": 123,
                    "contentId": 456,
                    "eventType": 0
                }
            ],
            "params": {
                "requestType": "user_recommendations",
                "numRecommendations": 5
            }
        },
        
        # Format 5: Try a completely different structure
        {
            "query": {
                "userId": 123,
                "count": 5
            }
        },
        
        # Format 6: Try a simple string
        "Get recommendations for user 123"
    ]
    
    # Try each request format
    for i, request_data in enumerate(request_formats):
        print(f"\n=== Testing Request Format {i+1} ===")
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        
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
            print("Response:")
            print(json.dumps(result_json, indent=2))
            
            # Check if we got any data
            if result_json and result_json != {"Results": {}}:
                print("Success! Got a non-empty response")
            else:
                print("Got an empty response")
        except Exception as e:
            print(f"Error: {e}")
            if isinstance(e, urllib.error.HTTPError):
                print(f"HTTP Error: {e.code}")
                print(f"Response: {e.read().decode('utf-8')}")

def test_azure_endpoint_with_swagger():
    """
    Try to access the Swagger documentation for the Azure ML endpoint
    """
    print("\n=== Testing Swagger Documentation ===")
    
    # Try different URLs that might have Swagger documentation
    swagger_urls = [
        'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/swagger.json',
        'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/swagger',
        'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/api-docs',
        'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/'
    ]
    
    for url in swagger_urls:
        print(f"\nTrying URL: {url}")
        
        try:
            # Set up headers
            headers = {
                'Accept': 'application/json'
            }
            
            # Create request
            req = urllib.request.Request(url, headers=headers)
            
            # Send request and get response
            response = urllib.request.urlopen(req)
            result = response.read()
            
            # Print the raw response
            print("Response:")
            print(result.decode('utf-8')[:500] + "..." if len(result) > 500 else result.decode('utf-8'))
            
            print("Success! Got a response")
        except Exception as e:
            print(f"Error: {e}")

def test_azure_endpoint_with_health_check():
    """
    Try to access health check endpoints for the Azure ML endpoint
    """
    print("\n=== Testing Health Check Endpoints ===")
    
    # Try different health check URLs
    health_urls = [
        'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/health',
        'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/status',
        'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/ping'
    ]
    
    for url in health_urls:
        print(f"\nTrying URL: {url}")
        
        try:
            # Set up headers
            headers = {
                'Accept': 'application/json'
            }
            
            # Create request
            req = urllib.request.Request(url, headers=headers)
            
            # Send request and get response
            response = urllib.request.urlopen(req)
            result = response.read()
            
            # Print the raw response
            print("Response:")
            print(result.decode('utf-8'))
            
            print("Success! Got a response")
        except Exception as e:
            print(f"Error: {e}")

def main():
    # Test the Azure ML endpoint with various request formats
    test_azure_endpoint()
    
    # Try to access the Swagger documentation
    test_azure_endpoint_with_swagger()
    
    # Try to access health check endpoints
    test_azure_endpoint_with_health_check()

if __name__ == "__main__":
    main()
