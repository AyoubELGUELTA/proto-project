# Utilisation d'une version LTS de Node (légère et stable)
FROM node:20-slim

# Installation des dépendances système minimales (si besoin de compiler certains modules)
RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie des fichiers de configuration NPM
COPY package*.json ./

# Installation des dépendances (y compris les devDependencies pour nodemon)
RUN npm install

# Copie du reste du code
COPY . .

# On expose le port du Gateway
EXPOSE 3000

# Utilisation de nodemon pour le "Hot Reload" (redémarrage auto lors des modifs)
# Note : Il faut que ton script "dev" dans package.json soit : "nodemon src/app.js"
CMD ["npm", "run", "dev"]