import express from 'express';
import { PythonShell } from 'python-shell';
import axios from 'axios';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { getValidIds } from '../models/get_valid_ids.js';

const router = express.Router();

// For ES modules to get __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Azure ML endpoint configuration
// Replace these with your actual Azure ML endpoint details
// Using placeholder values for testing
const AZURE_ML_ENDPOINT = 'https://placeholder-azure-ml-endpoint.azureml.net/api/v1/service/your-service-name/score';
const AZURE_ML_KEY = 'placeholder-api-key-for-testing';

// For testing purposes, we'll mock the Azure ML response
const mockAzureResponse = (id, isUser) => {
  return Array.from({ length: 5 }, (_, i) => ({
    contentId: `azure_item_${i + 300}`,
    score: (5 - i) / 5,
    reason: `Azure ML recommendation for ${isUser ? 'user' : 'item'} ${id}`
  }));
};

// Get recommendations from collaborative filtering model
router.get('/collaborative/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const isUser = req.query.type === 'user';
    
    // Options for PythonShell
    const options = {
      mode: 'json',
      pythonPath: 'python3', // Using python3 on macOS
      scriptPath: join(dirname(__dirname), 'models'),
      args: [id, isUser ? 'user' : 'item']
    };
    
    // Run the Python script to get recommendations from the collaborative filtering model
    PythonShell.run('collaborative_filtering.py', options).then(results => {
      if (results && results.length > 0) {
        res.json(results[0]);
      } else {
        res.status(404).json({ error: 'No recommendations found' });
      }
    }).catch(err => {
      console.error('Error running collaborative filtering script:', err);
      res.status(500).json({ error: 'Failed to get collaborative filtering recommendations' });
    });
  } catch (error) {
    console.error('Error in collaborative filtering endpoint:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get recommendations from content filtering model
router.get('/content/:id', async (req, res) => {
  try {
    const id = req.params.id;
    
    // Options for PythonShell
    const options = {
      mode: 'json',
      pythonPath: 'python3', // Using python3 on macOS
      scriptPath: join(dirname(__dirname), 'models'),
      args: [id]
    };
    
    // Run the Python script to get recommendations from the content filtering model
    PythonShell.run('content_filtering.py', options).then(results => {
      if (results && results.length > 0) {
        res.json(results[0]);
      } else {
        res.status(404).json({ error: 'No recommendations found' });
      }
    }).catch(err => {
      console.error('Error running content filtering script:', err);
      res.status(500).json({ error: 'Failed to get content filtering recommendations' });
    });
  } catch (error) {
    console.error('Error in content filtering endpoint:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get recommendations from Azure ML endpoint
router.get('/azure/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const isUser = req.query.type === 'user';
    
    // For testing purposes, use mock response instead of calling the actual endpoint
    // In production, uncomment the code below to call the real Azure ML endpoint
    /*
    // Prepare data for Azure ML endpoint
    const data = {
      Inputs: {
        input1: [
          {
            [isUser ? 'userId' : 'contentId']: id
          }
        ]
      },
      GlobalParameters: {}
    };
    
    // Call Azure ML endpoint
    const response = await axios.post(AZURE_ML_ENDPOINT, data, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${AZURE_ML_KEY}`
      }
    });
    
    if (response.data && response.data.Results) {
      // Extract and format recommendations from Azure ML response
      const recommendations = response.data.Results.output1.slice(0, 5);
      res.json(recommendations);
    } else {
      res.status(404).json({ error: 'No recommendations found from Azure ML' });
    }
    */
    
    // Use mock response for testing
    const mockRecommendations = mockAzureResponse(id, isUser);
    res.json(mockRecommendations);
    
  } catch (error) {
    console.error('Error calling Azure ML endpoint:', error);
    res.status(500).json({ error: 'Failed to get Azure ML recommendations' });
  }
});

// Get valid IDs for the dropdown
router.get('/valid-ids', async (req, res) => {
  try {
    const validIds = await getValidIds();
    res.json(validIds);
  } catch (error) {
    console.error('Error getting valid IDs:', error);
    res.status(500).json({ error: 'Failed to get valid IDs' });
  }
});

// Get all recommendations (from all three models)
router.get('/all/:id', async (req, res) => {
  try {
    const id = req.params.id;
    const isUser = req.query.type === 'user';
    
    // Make parallel requests to all three recommendation endpoints
    const [collaborativeResponse, contentResponse, azureResponse] = await Promise.all([
      axios.get(`http://localhost:${process.env.PORT || 3000}/api/recommendations/collaborative/${id}?type=${isUser ? 'user' : 'item'}`),
      axios.get(`http://localhost:${process.env.PORT || 3000}/api/recommendations/content/${id}`),
      axios.get(`http://localhost:${process.env.PORT || 3000}/api/recommendations/azure/${id}?type=${isUser ? 'user' : 'item'}`)
    ]);
    
    // Combine all recommendations
    const recommendations = {
      collaborative: collaborativeResponse.data,
      content: contentResponse.data,
      azure: azureResponse.data
    };
    
    res.json(recommendations);
  } catch (error) {
    console.error('Error getting all recommendations:', error);
    res.status(500).json({ error: 'Failed to get recommendations from all models' });
  }
});

export default router;
