import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [output, setOutput] = useState({ category1: [], category2: [], category3: [] });
  const [loading, setLoading] = useState(false);
  const [gradientPoints, setGradientPoints] = useState(
    Array.from({ length: 10 }, () => ({ x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight }))
  );

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

  useEffect(() => {
    const updateGradient = () => {
      setGradientPoints((points) =>
        points.map((point) => ({
          x: (point.x + (Math.random() * 0.5 - 0.25) + window.innerWidth) % window.innerWidth,
          y: (point.y + (Math.random() * 0.5 - 0.25) + window.innerHeight) % window.innerHeight,
        }))
      );
    };

    const handleMouseMove = (e) => {
      setGradientPoints((points) =>
        points.map((point) => {
          const distance = Math.hypot(point.x - e.clientX, point.y - e.clientY);
          if (distance < 100) {
            const factor = 0.05;
            return { 
              x: point.x + factor * (point.x - e.clientX), 
              y: point.y + factor * (point.y - e.clientY) 
            };
          }
          return point;
        })
      );
    };

    const interval = setInterval(updateGradient, 50);
    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      clearInterval(interval);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <div style={{ 
      position: 'relative',
      overflow: 'hidden',
      minHeight: '100vh',
      backgroundColor: '#000',
      color: '#fff',
      fontFamily: 'Arial',
      textAlign: 'center'
    }}>
      {gradientPoints.map((point, index) => (
        <div
          key={index}
          style={{
            position: 'absolute',
            left: point.x,
            top: point.y,
            width: '150px',
            height: '150px',
            background: 'radial-gradient(circle, rgba(80,0,80,0.6), rgba(70,0,70,0))',
            borderRadius: '50%',
            pointerEvents: 'none',
            transform: 'translate(-50%, -50%)',
            filter: 'blur(30px)',
          }}
        ></div>
      ))}
      <h1 style={{ 
        fontSize: '3.5rem', 
        fontWeight: 'bold', 
        marginBottom: '20px', 
        color: '#fff',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
        fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
      }}>
        apc.AI
      </h1>
      <button 
        onClick={runApplication} 
        disabled={loading} 
        style={{
          padding: '12px 24px',
          fontSize: '1.2rem',
          color: '#fff',
          backgroundColor: loading ? '#6c757d' : '#6200ea',
          border: 'none',
          borderRadius: '8px',
          cursor: loading ? 'not-allowed' : 'pointer',
          boxShadow: '0 6px 10px rgba(0, 0, 0, 0.15)',
          transition: 'background-color 0.3s ease, transform 0.2s ease',
          transform: loading ? 'none' : 'scale(1)',
        }}
        onMouseDown={(e) => !loading && (e.target.style.transform = 'scale(0.95)')}
        onMouseUp={(e) => (e.target.style.transform = 'scale(1)')}
      >
        {loading ? 'Running...' : 'Run Application'}
      </button>
      <div style={{ marginTop: '20px', width: '100%', maxWidth: '600px' }}>
        <h2 style={{ marginBottom: '15px', color: '#fff' }}>Outputs</h2>
        <div>
          <h3 style={{ color: '#28a745' }}>Good</h3>
          <ul style={{ listStyleType: 'none', padding: 0 }}>
            {output.category1.map((item, index) => (
              <li key={index} style={{ background: '#e9f7ef', padding: '5px 10px', margin: '5px 0', borderRadius: '3px' }}>
                {item}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 style={{ color: '#dc3545' }}>Bad</h3>
          <ul style={{ listStyleType: 'none', padding: 0 }}>
            {output.category2.map((item, index) => (
              <li key={index} style={{ background: '#f8d7da', padding: '5px 10px', margin: '5px 0', borderRadius: '3px' }}>
                {item}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 style={{ color: '#ffc107' }}>Review</h3>
          <ul style={{ listStyleType: 'none', padding: 0 }}>
            {output.category3.map((item, index) => (
              <li key={index} style={{ background: '#fff3cd', padding: '5px 10px', margin: '5px 0', borderRadius: '3px' }}>
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
