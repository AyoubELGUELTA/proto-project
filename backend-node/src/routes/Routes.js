const express = require('express');
const router = express.Router();
const multer = require('multer');
const ingestController = require('../controllers/ingestController');
const queryController = require('../controllers/queryController');
const documentController = require('../controllers/documentController');

// Configuration de Multer pour garder le fichier en mémoire vive
const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 15 * 1024 * 1024 } // Limite à 10 MB
});




// Middleware to secure our fastapi calls
const auth = require('../middlewares/auth');

// Route : POST /api/pdf/ingest
router.post('/ingest', auth, upload.single('file'), ingestController.processPdfUpload);

router.get('/query', auth, queryController.askQuestion);

router.post('/clear-history', queryController.resetHistory);

router.get('/documents', auth, documentController.getDocuments);


module.exports = router;