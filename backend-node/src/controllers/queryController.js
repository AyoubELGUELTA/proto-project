const pythonService = require('../services/pythonApiService');
const Joi = require('joi');

// Permet d'indiquer a l'utilisateur que sa requete doit faire un minimum de caracteres (3), et la limit du nombres de chunks aussi.

const querySchema = Joi.object({
    question: Joi.string().trim().min(3).max(500).required()
        .messages({
            'string.min': 'La question doit faire au moins 3 caractères.',
            'any.required': 'La question est obligatoire.'
        }),
    limit: Joi.number().integer().min(1).max(50).default(15)
});
exports.askQuestion = async (req, res) => {
    try {

        const { error, value } = querySchema.validate(req.query); // On valide le corps de la question avec joi

        if (error) {
        return res.status(400).json({ error: error.details[0].message });
        }

        const { question, limit } = value;

        console.log(`[Node] Question validée : "${question}" (limit: ${limit})`);
        
        const result = await pythonService.queryRAG(question, limit);
    
        res.status(200).json({
            status: "success",
            answer: result.answer,
            standalone_query: result.standalone_query || "pas de standalone query",
            sources: result.sources // le front aura besoin des sources pour l'UX
        });
    } catch (error) {
        console.error("[Node Error Query]", error.message);
        res.status(500).json({
            error: "Erreur lors de la récupération de la réponse du RAG.",
            details: error.message
        });
    }
};

exports.resetHistory = async (req, res) => {
    try {
        console.log(`[Node] Demande de réinitialisation de l'historique...`);
        
        const result = await pythonService.clearChatHistory();
        
        res.status(200).json({
            status: "success",
            message: "L'historique a été vidé avec succès.",
            data: result
        });
    } catch (error) {
        console.error("[Node Error Reset]", error.message);
        res.status(500).json({
            error: "Erreur lors de la réinitialisation de l'historique.",
            details: error.message
        });
    }
};