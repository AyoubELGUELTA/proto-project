const pythonService = require('../services/pythonApiService');

exports.processPdfUpload = async (req, res) => {
    try {
        // 1. Adaptation pour gérer req.file (single) ou req.files (multiple)
        const files = req.files || (req.file ? [req.file] : null);
        const config_id = req.body.config_id || "01"; // Récupère la config pour le benchmark

        if (!files || files.length === 0) {
            return res.status(400).json({ error: "Aucun fichier PDF fourni." });
        }

        console.log(`[Node] Réception de ${files.length} fichier(s). Config: ${config_id}`);

        // 2. Validation rapide du type MIME pour chaque fichier [cite: 2026-02-11]
        const invalidFiles = files.filter(f => f.mimetype !== 'application/pdf');
        if (invalidFiles.length > 0) {
            return res.status(400).json({ 
                error: "Certains fichiers ne sont pas des PDFs.",
                invalid: invalidFiles.map(f => f.originalname)
            });
        }

        console.log(`[Node] Envoi en masse vers FastAPI (Bulk Ingestion)...`);
        
        // 3. Appel au service mis à jour [cite: 2026-02-11]
        const result = await pythonService.sendBulkToPythonForIngestion(files, config_id);
        
        res.status(200).json({
            status: "success",
            message: `${files.length} PDF(s) traité(s), indexé(s) et digéré(s).`,
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