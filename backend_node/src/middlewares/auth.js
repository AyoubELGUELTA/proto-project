const jwt = require('jsonwebtoken');

module.exports = (req, res, next) => {
    try {
        const token = req.headers.authorization?.split(' ')[1]; // Format "Bearer TOKEN"
        if (!token) {
            return res.status(401).json({ error: "Accès refusé. Token manquant." });
        }

        // Pour le MVP, tu peux soit vérifier un vrai JWT, 
        // soit comparer avec une simple variable d'env (ADMIN_KEY)
        if (token !== process.env.JWT_SECRET) {
             return res.status(403).json({ error: "Token invalide. Acces refusé, désolé." });
        }
        
        next();
    } catch (error) {
        res.status(401).json({ error: "Requête non authentifiée." });
    }
};