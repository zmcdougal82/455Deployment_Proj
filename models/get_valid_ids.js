import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { PythonShell } from 'python-shell';
import fs from 'fs';

// For ES modules to get __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Function to get valid IDs from the models
export async function getValidIds() {
  return new Promise((resolve, reject) => {
    // Options for PythonShell
    const options = {
      mode: 'json',
      pythonPath: 'python3', // Using python3 on macOS
      scriptPath: __dirname
    };
    
    // Run the Python script to get valid IDs
    PythonShell.run('get_valid_ids.py', options)
      .then(results => {
        if (results && results.length > 0) {
          resolve(results[0]);
        } else {
          reject(new Error('No valid IDs found'));
        }
      })
      .catch(err => {
        console.error('Error getting valid IDs:', err);
        reject(err);
      });
  });
}
