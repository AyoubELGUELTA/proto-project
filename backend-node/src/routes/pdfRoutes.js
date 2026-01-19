const express = require('express');
const router = express.Router();
const multer = require('multer');
const pdfController = require('../controllers/pdfIngestController');

// Configuration de Multer pour garder le fichier en mémoire vive
const upload = multer({ storage: multer.memoryStorage() });

// Route : POST /api/pdf/ingest
router.post('/ingest', upload.single('file'), pdfController.processPdfUpload);

module.exports = router;