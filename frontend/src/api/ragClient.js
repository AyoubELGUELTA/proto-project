import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_NODE_API_URL,
  headers: {
    'Authorization': `Bearer ${import.meta.env.VITE_NODE_API_TOKEN}`
  }
});

export const ragApi = {
  // Upload du PDF
  ingestPdf: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/ingest', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  // Pose une question
  askQuestion: (question) => api.get('/query', { params: { question } }),
  // Reset l'historique
  clearHistory: () => api.post('/clear-history'), 
  getIngestedDocuments: () => api.get('/documents'),
};