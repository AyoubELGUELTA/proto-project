require('dotenv').config();
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');

const app = express();

// Base middlewares
app.use(helmet()); // Security
app.use(cors());   // Authorize futur frontend
app.use(express.json()); // Read Json send to server
app.use(morgan('dev')); // Logs of requests

// Test route
app.get('/health', (req, res) => {
    res.status(200).json({ status: 'Node Backend is running' });
});

// Importation des routes (à créer après)
// app.use('/api/auth', require('./routes/authRoutes'));
// app.use('/api/chat', require('./routes/chatRoutes'));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Backend running on port ${PORT}`);
});