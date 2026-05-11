const pythonService = require('../services/pythonApiService');


exports.getDocuments = async (req, res) => {
    try {
        // AJOUT DES PARENTHÈSES ICI : getDocuments()
        const documents = await pythonService.getDocuments(); 
        
        // Log pour vérifier ce que Node reçoit de Python avant d'envoyer au Front
        console.log("Documents de Python:", documents);
        
        res.status(200).json({ documents: documents });
    } catch (error) {
        console.error("Erreur contrôleur Node:", error);
        res.status(500).json({ error: "Impossible de récupérer la liste des documents" });
    }
};