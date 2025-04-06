#!/usr/bin/env python3
import requests
import json
import sys

def test_azure_integration():
    """
    Test the Azure ML integration in the news recommender app
    """
    # Base URL for the API
    base_url = "http://localhost:3001/api/recommendations"
    
    # Test IDs
    test_ids = [
        "1234567890",  # Regular integer ID
        "6756039155228175109",  # ID from local models
        "-3933783680725097100"  # Negative ID
    ]
    
    for test_id in test_ids:
        print(f"\n=== Testing with ID: {test_id} ===")
        
        # Test Azure endpoint directly
        print("\nTesting Azure endpoint directly:")
        azure_url = f"{base_url}/azure/{test_id}?type=user"
        
        try:
            response = requests.get(azure_url)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Got {len(result)} recommendations")
                
                # Print the first recommendation
                if len(result) > 0:
                    print("\nFirst recommendation:")
                    print(f"Content ID: {result[0]['contentId']}")
                    print(f"Score: {result[0]['score']}")
                    print(f"Reason: {result[0]['reason']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test all recommendations endpoint
        print("\nTesting all recommendations endpoint:")
        all_url = f"{base_url}/all/{test_id}?type=user"
        
        try:
            response = requests.get(all_url)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if we got Azure recommendations
                if "azure" in result and len(result["azure"]) > 0:
                    print(f"Got {len(result['azure'])} Azure recommendations")
                    
                    # Print the first Azure recommendation
                    print("\nFirst Azure recommendation:")
                    print(f"Content ID: {result['azure'][0]['contentId']}")
                    print(f"Score: {result['azure'][0]['score']}")
                    print(f"Reason: {result['azure'][0]['reason']}")
                else:
                    print("No Azure recommendations found")
                
                # Check if we got collaborative recommendations
                if "collaborative" in result and len(result["collaborative"]) > 0:
                    print(f"\nGot {len(result['collaborative'])} collaborative recommendations")
                else:
                    print("\nNo collaborative recommendations found")
                
                # Check if we got content recommendations
                if "content" in result and len(result["content"]) > 0:
                    print(f"\nGot {len(result['content'])} content recommendations")
                else:
                    print("\nNo content recommendations found")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("=== Testing Azure ML Integration ===")
    print("Make sure the news recommender app is running on port 3001")
    print("You can start it with: npm start")
    
    # Ask for confirmation
    input("Press Enter to continue...")
    
    test_azure_integration()

if __name__ == "__main__":
    main()
