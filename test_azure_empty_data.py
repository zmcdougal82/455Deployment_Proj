#!/usr/bin/env python3
import urllib.request
import json

def test_azure_with_empty_data():
    """
    Test the Azure ML endpoint with an empty data object and the new API key
    """
    # Empty data object
    data = {}
    
    # Convert data to JSON string and then to bytes
    body = str.encode(json.dumps(data))
    
    # Azure ML endpoint URL
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    
    # New API key provided by the user
    api_key = 'Ydnjdd8tr1wwAtpCJG7bftYOQYPXxO2j'
    
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")
    
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
        
        # Print the raw response
        print("Raw response:")
        print(result)
        
        # Try to parse as JSON
        try:
            result_json = json.loads(result)
            print("\nJSON response:")
            print(json.dumps(result_json, indent=2))
        except:
            print("\nResponse is not valid JSON")
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def test_azure_with_structured_data_new_key():
    """
    Test the Azure ML endpoint with structured data and the new API key
    """
    # Structured data based on Swagger documentation
    data = {
        "Inputs": {
            "input1": [
                {
                    "personId": 1908339160857512799,
                    "contentId": 3460026829794173084,
                    "eventType": 1
                }
            ]
        },
        "GlobalParameters": {
            "requestType": "user_recommendations",
            "numRecommendations": 5
        }
    }
    
    # Convert data to JSON string and then to bytes
    body = str.encode(json.dumps(data))
    
    # Azure ML endpoint URL
    url = 'http://04c0d1c8-3578-4221-a4ab-16aaabbc1d66.eastus2.azurecontainer.io/score'
    
    # New API key provided by the user
    api_key = 'Ydnjdd8tr1wwAtpCJG7bftYOQYPXxO2j'
    
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")
    
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
        
        # Print the raw response
        print("\nRaw response:")
        print(result)
        
        # Try to parse as JSON
        try:
            result_json = json.loads(result)
            print("\nJSON response:")
            print(json.dumps(result_json, indent=2))
        except:
            print("\nResponse is not valid JSON")
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def main():
    print("=== Testing with empty data and new API key ===")
    test_azure_with_empty_data()
    
    print("\n=== Testing with structured data and new API key ===")
    test_azure_with_structured_data_new_key()

if __name__ == "__main__":
    main()
