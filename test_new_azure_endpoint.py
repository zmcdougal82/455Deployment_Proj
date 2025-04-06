#!/usr/bin/env python3
import requests
import json
import sys

def test_new_azure_endpoint():
    """
    Test the new Azure ML endpoint with the provided code
    """
    url = "http://64db7000-54c2-42d1-b823-623f999523bb.eastus2.azurecontainer.io/score"
    key = "q8sI2j8rSq5ctlieaQtUAHsHxqMb7OA6"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    
    data = {
        "Inputs": {
            "input1": [
                {
                    "personId": -8.845298781299428e+18
                },
                {
                    "personId": -1.0320192293846964e+18
                },
                {
                    "personId": -1.1302722942469832e+18
                }
            ]
        }
    }
    
    print("Request data:")
    print(json.dumps(data, indent=2))
    
    try:
        # Send request and get response
        response = requests.post(url, headers=headers, data=str.encode(json.dumps(data)))
        
        # Print status code
        print(f"\nStatus code: {response.status_code}")
        
        # Print raw response
        print("\nRaw response:")
        print(response.text)
        
        # Try to parse as JSON
        try:
            result_json = json.loads(response.text)
            print("\nJSON response:")
            print(json.dumps(result_json, indent=2))
            
            # Check if we got the expected structure
            if 'Results' in result_json and 'WebServiceOutput0' in result_json['Results']:
                print("\nSuccess! Got the expected response structure")
                
                # Print key-value pairs from the first item
                print("\nKey-value pairs from the first item:")
                for k, v in result_json['Results']['WebServiceOutput0'][0].items():
                    print(f"{k}: {v}")
            else:
                print("\nResponse doesn't have the expected structure")
        except json.JSONDecodeError:
            print("\nResponse is not valid JSON")
    except Exception as e:
        print(f"\nError: {e}")

def test_with_different_ids():
    """
    Test the new Azure ML endpoint with different IDs
    """
    url = "http://64db7000-54c2-42d1-b823-623f999523bb.eastus2.azurecontainer.io/score"
    key = "q8sI2j8rSq5ctlieaQtUAHsHxqMb7OA6"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    
    # Try with different IDs
    test_cases = [
        # Test case 1: Single ID
        {
            "Inputs": {
                "input1": [
                    {
                        "personId": -8.845298781299428e+18
                    }
                ]
            }
        },
        
        # Test case 2: Different format for IDs (regular integers)
        {
            "Inputs": {
                "input1": [
                    {
                        "personId": 1234567890
                    },
                    {
                        "personId": 9876543210
                    }
                ]
            }
        },
        
        # Test case 3: IDs from our local models
        {
            "Inputs": {
                "input1": [
                    {
                        "personId": 6756039155228175109
                    },
                    {
                        "personId": -3933783680725097100
                    },
                    {
                        "personId": 8195788452563155020
                    }
                ]
            }
        }
    ]
    
    for i, test_data in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        print("Request data:")
        print(json.dumps(test_data, indent=2))
        
        try:
            # Send request and get response
            response = requests.post(url, headers=headers, data=str.encode(json.dumps(test_data)))
            
            # Print status code
            print(f"\nStatus code: {response.status_code}")
            
            # Try to parse as JSON
            try:
                result_json = json.loads(response.text)
                print("\nJSON response:")
                print(json.dumps(result_json, indent=2))
                
                # Check if we got the expected structure
                if 'Results' in result_json and 'WebServiceOutput0' in result_json['Results']:
                    print("\nSuccess! Got the expected response structure")
                else:
                    print("\nResponse doesn't have the expected structure")
            except json.JSONDecodeError:
                print("\nResponse is not valid JSON")
                print("Raw response:")
                print(response.text)
        except Exception as e:
            print(f"\nError: {e}")

def main():
    print("=== Testing New Azure ML Endpoint ===")
    test_new_azure_endpoint()
    
    print("\n=== Testing with Different IDs ===")
    test_with_different_ids()

if __name__ == "__main__":
    main()
