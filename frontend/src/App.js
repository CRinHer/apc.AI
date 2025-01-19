import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [output, setOutput] = useState({ category1: [], category2: [], category3: [] });
  const [loading, setLoading] = useState(false);

  const runApplication = async () => {
    setLoading(true);
    try {
      // Replace with your Python application's API endpoint
      const response = await axios.get('http://localhost:5000/run');
      setOutput(response.data);
    } catch (error) {
      console.error('Error running the application:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>apc.AI</h1>
      <button onClick={runApplication} disabled={loading}>
        {loading ? 'Running...' : 'Run Application'}
      </button>
      <div style={{ marginTop: '20px' }}>
        <h2>Outputs</h2>
        <div>
          <h3>Good</h3>
          <ul>
            {output.category1.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Bad</h3>
          <ul>
            {output.category2.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Review</h3>
          <ul>
            {output.category3.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
