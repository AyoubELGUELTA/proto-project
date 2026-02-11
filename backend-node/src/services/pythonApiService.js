const axios = require('axios');
const FormData = require('form-data');

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://fastapi_rag:8000';

/* Envoie un fichier PDF au moteur RAG Python pour traitement et stockage vectoriel.*/
exports.sendBulkToPythonForIngestion = async (files, config_id = "01") => {
    try {
        const form = new FormData();
        
        // On boucle sur le tableau de fichiers (venant de multer)
        files.forEach((file) => {
            // Important : on utilise la clé 'files' (au pluriel) pour correspondre à l'argument FastAPI
            form.append('files', file.buffer, {
                filename: file.originalname,
                contentType: file.mimetype,
            });
        });

        // On ajoute le paramètre de configuration pour le benchmark
        form.append('config_id', config_id);

        const response = await axios.post(`${FASTAPI_URL}/ingest-bulk`, form, {
            headers: { ...form.getHeaders() },
            timeout: 1000000 //16m 40s
        });

        return response.data;
    } catch (error) {
        console.error("❌ Node Gateway Bulk Error:", error.message);
        throw new Error(error.response?.data?.detail || error.message);
    }
};

/* Interroge le moteur RAG pour obtenir une réponse basée sur les documents indexés */
exports.queryRAG = async (question, limit = 20) => {
    try {
        const response = await axios.get(`${FASTAPI_URL}/query`, {
            params: {
                question: question,
                limit: limit
            },
            timeout: 90000 // 1m 30
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.detail || error.message);
    }
};


exports.clearChatHistory = async () => {
    try {
        const response = await axios.post(`${FASTAPI_URL}/clear-history`, {}, {
            timeout: 15000 //rapide, car on purge une liste
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.detail || error.message);
    }
};

exports.getDocuments = async () => {
    try {
    const response = await axios.get(`${FASTAPI_URL}/ingested-documents`, {}, {
            timeout: 5000 //rapide, car on récupere une liste
        });
    return response.data.documents;

}
catch (error) {
        throw new Error(error.response?.data?.detail || error.message);
    }
}