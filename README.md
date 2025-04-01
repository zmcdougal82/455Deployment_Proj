# News Article Recommender

A web application that recommends news articles using multiple recommendation models:
1. Collaborative Filtering (Python model)
2. Content Filtering (Python model)
3. Wide and Deep Recommender (Azure ML)

## Project Overview

This application allows users to select a user ID or content ID and get recommendations from three different recommendation models. The recommendations are displayed side by side for comparison.

## Prerequisites

- Node.js (v14 or higher)
- Python (v3.6 or higher)
- Azure ML Studio account with a deployed Wide and Deep Recommender endpoint

## Setup Instructions

### 1. Install Node.js Dependencies

```bash
cd news_recommender
npm install
```

### 2. Install Python Dependencies

```bash
pip install numpy scikit-learn pandas pickle-mixin
```

### 3. Configure Azure ML Endpoint

Edit the `routes/recommendations.js` file and update the Azure ML endpoint configuration:

```javascript
// Azure ML endpoint configuration
const AZURE_ML_ENDPOINT = 'YOUR_AZURE_ML_ENDPOINT_URL';
const AZURE_ML_KEY = 'YOUR_AZURE_ML_API_KEY';
```

### 4. Place Model Files

Place your trained model files in the `models` directory:
- `collaborative_model.sav` - Your collaborative filtering model
- `content_model.sav` - Your content filtering model

### 5. Start the Application

```bash
npm start
```

The application will be available at http://localhost:3000

## Usage

1. Open the application in your web browser
2. Select whether you want to search by User ID or Content ID
3. Enter the ID value
4. Click "Get Recommendations"
5. View the recommendations from all three models

## Project Structure

```
news_recommender/
├── public/                 # Static files
│   └── index.html          # Main HTML page
├── models/                 # Python model scripts and saved models
│   ├── collaborative_filtering.py
│   ├── content_filtering.py
│   ├── collaborative_model.sav  # Your saved collaborative model
│   └── content_model.sav        # Your saved content model
├── routes/                 # Express routes
│   └── recommendations.js  # API endpoints for recommendations
├── app.js                  # Main application file
├── package.json            # Node.js dependencies
└── README.md               # This file
```

## Customization

### Adapting the Python Scripts

The Python scripts in the `models` directory contain placeholder implementations. You'll need to adapt them to work with your specific models:

1. Update the `load_model` function if your model requires special loading logic
2. Modify the `get_recommendations` function to use your model's API for generating recommendations

### Modifying the Frontend

The frontend is a simple HTML/CSS/JavaScript application. You can customize it by editing the `public/index.html` file.

## Notes

- The application uses the `python-shell` package to execute Python scripts from Node.js
- The Python scripts should return recommendations as JSON to be parsed by the Node.js application
- Make sure your Python environment has all the necessary dependencies installed
# 455Deployment_Proj
