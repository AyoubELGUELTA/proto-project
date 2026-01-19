const pythonService = require('../services/pythonApiService');

exports.processPdfUpload = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "Aucun fichier PDF fourni." });
        }
        console.log("Fichier reçu :", req.file); // DEBUG
        // Vérification rapide du type MIME pour la sécurité
        if (req.file.mimetype !== 'application/pdf') {
            return res.status(400).json({ error: "Le fichier doit être un PDF." });
        }

        console.log(`[Node] Envoi du PDF ${req.file.originalname} vers FastAPI pour le garder en BDD...`);
        
        const result = await pythonService.sendToPythonForPdfIngestion(req.file);
        
        res.status(200).json({
            status: "success",
            message: "Le PDF a été traité, indexé et digéré.",
            data: result
        });
    } catch (error) {
        console.error("[Node Error]", error.message);
        res.status(500).json({
            error: "Erreur lors de la communication avec le moteur d'ingestion PDF.",
            details: error.message
        });
    }
};