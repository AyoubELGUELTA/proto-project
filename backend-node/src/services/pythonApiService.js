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
        timeout: 300000 // 5 minutes pour les gros PDF

    });
    return response.data;
};