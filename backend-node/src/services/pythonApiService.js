const axios = require('axios');
const FormData = require('form-data');

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://fastapi_rag:8000';

/* Envoie un fichier PDF au moteur RAG Python pour traitement et stockage vectoriel.*/
exports.sendToPythonForPdfIngestion = async (file) => {
    
    const form = new FormData();
    // On repasse le fichier reçu à FastAPI
    form.append('file', file.buffer, {
        filename: file.originalname,
        contentType: file.mimetype,
    });

    const response = await axios.post(`${FASTAPI_URL}/ingest_pdf`, 
        form, 
        {headers: { ...form.getHeaders() },
        timeout: 10000000 // 5 minutes pour les gros PDF

    });
    return response.data;
};

/* Interroge le moteur RAG pour obtenir une réponse basée sur les documents indexés */
exports.queryRAG = async (question, limit = 15) => {
    try {
        const response = await axios.get(`${FASTAPI_URL}/query`, {
            params: {
                question: question,
                limit: limit
            },
            timeout: 60000 // 1 minute suffit généralement pour une génération de réponse
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