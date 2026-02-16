# 🚀 Pivot Architectural : De RAG Vectoriel à Entity-Centric RAG

### 📅 Date : 16 Février 2026
### 🎯 État des lieux & Décision
Après analyse des limites du RAG purement vectoriel (perte d'exhaustivité sur les synthèses d'entités), la décision est prise de pivoter vers une architecture **Entity-Centric**. L'objectif est de transformer un moteur de recherche par similarité en un système de connaissance structuré ("Maître Virtuel").

### 🏗️ Nouvelle Stratégie Technique
1. **Unification Database** : Migration de Qdrant vers **PostgreSQL + pgvector** pour centraliser les relations et les vecteurs.
2. **Extraction d'Entités** : Implémentation d'un pipeline d'extraction (LLM-based) lors de l'ingestion pour identifier Personnages, Lieux et Concepts.
3. **Résolution d'Entités** : Utilisation d'un système d'**Aliases** (TEXT ARRAY) pour gérer les variantes orthographiques et phonétiques.
4. **Taxonomie Souple** : Système de tagging thématique (Tags) pour permettre un filtrage hybride (ex: Entité "Hajj" + Tag "Jurisprudence").
5. **Importance Dynamique** : Le poids des entités sera calculé par la densité de leurs liens (`entity_links`) plutôt que par une classification manuelle.

### 🚩 Prochaines Étapes (Sprint 1)
- [ ] Initialisation de la nouvelle base PostgreSQL avec pgvector.
- [ ] Création du script de migration du schéma DDL.
- [ ] Développement du pipeline d'extraction d'entités avec GPT-4o-mini.
- [ ] Test d'ingestion sur le corpus "Mères des Croyants".

"Make it work, then make it work well. Today, we build the foundation."
