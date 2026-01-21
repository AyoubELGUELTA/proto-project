require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');
const axios = require('axios');

const app = express();

// Base middlewares
app.use(helmet()); // Security
app.use(cors());   // Authorize future frontend
app.use(express.json()); // Read Json send to server
app.use(morgan('dev')); // Logs of requests


// --- Configuration des URLs ---
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://fastapi_rag:8000';

// Allows our frontend to communicate with nodegateway
app.use(cors({
    origin: process.env.FRONTEND_URL || 'http://localhost:5173',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// --- Routes de monitoring ---
app.get('/test-python', async (req, res) => { // Test pour voir si les containers backends peuvent commumiquer dans le docker compose
  try {
    // Le Node demande au Python s'il est réveillé
    const response = await axios.get(`${FASTAPI_URL}/`);
    res.json({
      message: "Le Gateway Node a réussi à parler au Python !",
      python_response: response.data
    });
  } catch (error) {
    res.status(500).json({
      message: "Erreur de communication avec le Python",
      details: error.message
    });
  }
});
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'Node Backend is running' });
});

// --- Importation des routes fonctionnelles ---
app.use('/api', require('./routes/Routes'));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Backend running on port ${PORT} ;)`);
});