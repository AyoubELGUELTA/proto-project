"""
Prompts pour extraction Graph RAG - Domaine Sira
Structure modulaire préparée pour multi-domaines futurs
"""

from typing import Dict

# ============================================================
# DOMAIN: SIRA (Prophetic Biography)
# ============================================================

SIRA_ENTITY_PASS1 = """
Tu es un expert en extraction d'entités pour un Knowledge Graph Islamique spécialisé en SIRA (Biographie Prophétique).

CONTEXTE DU DOCUMENT:
{identity_context}

CHUNK À ANALYSER:
{chunk_text}

ONTOLOGIE - TYPES D'ENTITÉS SIRA:

1. **Prophet** (Prophète)
   - Le Prophète Muhammad (ﷺ)
   - Toujours avec honorific: SAW, ﷺ, صلى الله عليه وسلم

2. **MotherBeliever** (Mères des Croyants - أمهات المؤمنين)
   - Les 11 épouses du Prophète (ﷺ):
     * Khadija bint Khuwaylid
     * Sawda bint Zam'a
     * Aïcha bint Abi Bakr
     * Hafsa bint Umar
     * Zaynab bint Khuzayma
     * Umm Salama (Hind bint Abi Umayya)
     * Zaynab bint Jahsh
     * Juwayriya bint al-Harith
     * Umm Habiba (Ramla bint Abi Sufyan)
     * Safiyya bint Huyayy
     * Maymuna bint al-Harith
   - Format: Nom complet avec filiation (bint = fille de)
   - Toujours avec honorific: RA, رضي الله عنها
   ⚠️ CONTRE-EXEMPLES (PAS MotherBeliever):
   - Fatima bint Muhammad → AhlBayt (fille, pas épouse)
   - Asma bint Abi Bakr → Sahabiya (sœur d'Aïcha, pas épouse Prophète)
   - Umm Kulthum bint Muhammad → AhlBayt (fille)

3. **Sahabi** (Compagnons - الصحابة)
   - Hommes ayant rencontré le Prophète (ﷺ) et cru en lui
   - Exemples: Abu Bakr, Umar ibn al-Khattab, Uthman, Ali, Bilal, Salman al-Farsi
   - Honorific: RA, رضي الله عنه
   
4. **Sahabiya** (Compagnons féminins)
   - Femmes compagnons (hors Mères des Croyants)
   - Exemples: Asma bint Abi Bakr, Umm Sulaym, Fatima bint Muhammad
   - Honorific: RA, رضي الله عنها

5. **AhlBayt** (Famille du Prophète - أهل البيت)
   - Famille directe: Fatima, Hassan, Hussain, Ali ibn Abi Talib
   - Oncles: Abu Talib, Hamza, Abbas
   - Tantes: Safiya bint Abdul-Muttalib

6. **Tribe** (Tribus arabes - القبائل)
   - Quraysh (sous-clans: Banu Hashim, Banu Umayya, Banu Makhzum)
   - Aws, Khazraj (tribus de Médine)
   - Banu Nadir, Banu Qaynuqa, Banu Qurayza (tribus juives)
   - Thaqif, Hawazin, Ghatafan

7. **Place** (Lieux - الأماكن)
   - Villes: Mecca (مكة), Medina (المدينة), Ta'if, Abyssinia
   - Sites sacrés: Ka'ba, Masjid al-Haram, Masjid Nabawi
   - Champs de bataille: Badr, Uhud, Khandaq, Khaybar, Hunayn
   - Montagnes: Hira, Thawr, Uhud

8. **Battle** (Batailles - الغزوات)
   - Ghazawat (dirigées par le Prophète): Badr, Uhud, Khandaq, Khaybar, Mecca Conquest, Hunayn, Tabuk
   - Saraya (expéditions sans le Prophète): Nakhla, Mu'tah
   - Date format: Année Hijri (ex: 2 AH, 5 AH)

9. **Event** (Événements historiques - الأحداث)
   - Révélation première (610 CE)
   - Hijra (Migration à Médine, 622 CE / 1 AH)
   - Traités: Hudaybiya, Aqaba
   - Mariages prophétiques
   - Naissances/Décès de figures clés

10. **Period** (Périodes - الفترات)
    - Meccan Period (Période Mecquoise, 610-622 CE)
    - Medinan Period (Période Médinoise, 622-632 CE)
    - Pre-Hijra, Post-Hijra
    - Contexte: Jahiliya (pré-islamique)

11. **Concept** (Concepts religieux/historiques)
    - Hijra, Shahada, Bay'ah (serment d'allégeance)
    - Ansar (Auxiliaires de Médine), Muhajirun (Émigrés)
    - Ahl al-Kitab (Gens du Livre)

RÈGLES D'EXTRACTION CRITIQUES:

1. **Noms complets obligatoires**
   - ✅ "Aïcha bint Abi Bakr" (nom + filiation)
   - ❌ "Aïcha" seul (incomplet)
   - ✅ "Umar ibn al-Khattab"
   - ❌ "Umar" seul

2. **Honorifics obligatoires**
   - Prophète: SAW, ﷺ
   - Mères/Compagnons: RA, رضي الله عنه/ها
   - Si absent dans texte, AJOUTER dans normalized_name

3. **Variantes orthographiques (aliases)**
   ✅ INCLURE:
   - Noms complets variantes: ["Aïcha bint Abi Bakr", "A'isha bint Abu Bakr"]
   - Noms arabes: ["عائشة بنت أبي بكر"]
   - Surnoms UNIQUES: ["al-Siddiqah"] (la Véridique)
   - Kunya SI UNIQUE (ou QUASIMENT): ["Umm Abdullah"] (mère d'Abdullah)
   
   ❌ NE PAS INCLURE:
   - Prénoms seuls: "Aïcha", "Abdullah" (trop communs, non-distinctifs)
   - Titres génériques: "Mère des Croyants" (s'applique à 11 personnes)
   - Articles seuls: "al-", "ibn", "bint"
   
   🎯 RÈGLE D'OR: Alias doit identifier UNIQUEMENT cette personne
   
   Exemples:
   ✅ "Abu Bakr al-Siddiq" → aliases: ["Abdullah ibn Abi Quhafa", "al-Siddiq"]
   ❌ PAS: ["Abu Bakr", "Abdullah"] (trop communs)
   
   ✅ "Aïcha bint Abi Bakr" → aliases: ["A'isha bint Abu Bakr", "Umm Abdullah"]
   ❌ PAS: ["Aïcha", "عائشة"] seuls

4. **Disambiguation par contexte**
   - "Zaynab" seule = ambigu (2 Mères portent ce nom)
   - → Zaynab bint Khuzayma vs Zaynab bint Jahsh
   - Utilise contexte (tribu, événement, date) pour identifier

5. **Tribus: Format précis**
   - "Banu Hashim" (pas juste "Hashim")
   - "Quraysh" (tribu mère)
   - Sous-clans explicites si mentionnés

6. **Dates: Conversion Hijri**
   - Si texte dit "2 ans après Hijra" → "2 AH"
   - Si date grégorienne → ajouter équivalent Hijri si possible

FORMAT DE SORTIE JSON:

{{
    "entities": [
        {{
            "name": "Nom complet canonique",
            "normalized_name": "Version normalisée (lowercase, sans accents, avec honorific)",
            "type": "Prophet|MotherBeliever|Sahabi|Sahabiya|AhlBayt|Tribe|Place|Battle|Event|Period|Concept",
            "aliases": ["Variante1", "نسخة عربية", "Transliteration"],
            "context_description": "Pourquoi cette entité dans ce chunk (1 phrase)",
            "confidence": 0.0-1.0
        }}
    ]
}}

EXEMPLES CONCRETS:

Texte: "Aïcha, la fille d'Abu Bakr, a participé à la bataille d'Uhud en apportant de l'eau."

Extraction:
{{
    "entities": [
        {{
            "name": "Aïcha bint Abi Bakr",
            "normalized_name": "aicha bint abi bakr ra",
            "type": "MotherBeliever",
            "aliases": ["Aïcha", "Aicha", "A'isha", "عائشة بنت أبي بكر"],
            "context_description": "Mère des Croyants, fille d'Abu Bakr, a participé à Uhud",
            "confidence": 1.0
        }},
        {{
            "name": "Abu Bakr al-Siddiq",
            "normalized_name": "abu bakr al siddiq ra",
            "type": "Sahabi",
            "aliases": ["Abu Bakr", "أبو بكر الصديق", "Abdullah ibn Abi Quhafa"],
            "context_description": "Père d'Aïcha, premier calife",
            "confidence": 1.0
        }},
        {{
            "name": "Battle of Uhud",
            "normalized_name": "battle of uhud",
            "type": "Battle",
            "aliases": ["Uhud", "غزوة أحد", "Ghazwat Uhud"],
            "context_description": "Bataille où Aïcha a apporté de l'eau aux combattants",
            "confidence": 1.0
        }}
    ]
}}

COMMENCE L'EXTRACTION (JSON uniquement, pas de commentaire):
"""

# ============================================================

SIRA_ENTITY_PASS2_GLEANING = """
Tu es un expert en REVIEW d'extraction d'entités pour la Sira.

CONTEXTE DU DOCUMENT:
{identity_context}

CHUNK ANALYSÉ:
{chunk_text}

ENTITÉS DÉJÀ EXTRAITES (Pass 1):
{entities_pass1}

MISSION: Trouve les entités RATÉES lors du Pass 1.

ENTITÉS SOUVENT RATÉES:

1. **Personnes mentionnées indirectement**
   - "La fille du Prophète" → Fatima bint Muhammad
   - "L'épouse d'Ali" → Fatima bint Muhammad
   - "Le fils de Khadija" → Qasim ibn Muhammad (décédé enfant)

2. **Lieux implicites**
   - "La ville" (contexte Médine) → Medina
   - "La maison" (si chez Prophète) → House of Prophet

3. **Tribus via nisba (affiliation)**
   - "al-Qurashi" → Tribe: Quraysh
   - "al-Ansari" → Ansar (Helpers of Medina)

4. **Événements non-nommés mais décrits**
   - "Quand ils ont migré" → Hijra
   - "Le jour de la grande victoire" → Battle of Badr

5. **Concepts implicites**
   - "Ceux qui ont émigré" → Muhajirun
   - "Les auxiliaires" → Ansar

6. **Variantes de noms déjà extraites**
   - Pass 1 a "Aïcha", texte dit aussi "Umm Abdullah" (surnom d'Aïcha)
   - → Ajouter alias, PAS nouvelle entité

RÈGLES GLEANING:

- ✅ Ajoute entité SI nouvelle personne/lieu/événement
- ❌ N'ajoute PAS si juste variante nom déjà extrait
- ✅ Ajoute si mention indirecte claire ("la fille du Prophète")
- ❌ N'invente PAS d'entité si contexte ambigu

FORMAT SORTIE JSON:

{{
    "additional_entities": [
        {{
            "name": "Nom complet",
            "normalized_name": "version normalisée",
            "type": "Type",
            "aliases": ["variantes"],
            "context_description": "Pourquoi ratée en Pass 1",
            "confidence": 0.0-1.0
        }}
    ],
    "rejected_candidates": [
        {{
            "candidate": "Nom candidat",
            "reason": "Pourquoi rejeté (doublon/ambigu/etc)"
        }}
    ]
}}

Si AUCUNE entité ratée, retourne:
{{
    "additional_entities": [],
    "rejected_candidates": []
}}

COMMENCE LA REVIEW (JSON uniquement):
"""

# ============================================================

SIRA_RELATION_PASS1 = """
Tu es un expert en extraction de RELATIONS pour un Knowledge Graph Sira.

CONTEXTE DU DOCUMENT:
{identity_context}

CHUNK ANALYSÉ:
{chunk_text}

ENTITÉS VALIDÉES DANS CE CHUNK:
{final_entities}

TYPES DE RELATIONS SIRA:

**FAMILIALES (أنساب)**

1. MARRIED_TO (متزوج من)
   - (Prophète)-[:MARRIED_TO]->(Mère des Croyants)
   - Propriétés: {{date_hijri: "3 AH", location: "Medina"}}

2. DAUGHTER_OF / SON_OF (ابنة / ابن)
   - (Aïcha)-[:DAUGHTER_OF]->(Abu Bakr)
   - (Fatima)-[:DAUGHTER_OF]->(Prophète)

3. MOTHER_OF / FATHER_OF (أم / أب)
   - (Khadija)-[:MOTHER_OF]->(Fatima)
   - (Prophète)-[:FATHER_OF]->(Ibrahim)

4. SIBLING (أخ / أخت)
   - (Aïcha)-[:SIBLING]->(Asma bint Abi Bakr)
   - Propriétés: {{type: "sister"}}

5. UNCLE_OF / AUNT_OF (عم / عمة)
   - (Hamza)-[:UNCLE_OF]->(Prophète)
   - (Safiya)-[:AUNT_OF]->(Prophète)

**TRIBALES (قبائل)**

6. MEMBER_OF_TRIBE (عضو قبيلة)
   - (Abu Bakr)-[:MEMBER_OF_TRIBE]->(Quraysh)
   - (Aïcha)-[:MEMBER_OF_TRIBE]->(Banu Taym) [sous-clan Quraysh]
   - Propriétés: {{clan: "Banu Hashim"}} si applicable

**ÉVÉNEMENTIELLES (أحداث)**

7. PARTICIPATED_IN (شارك في)
   - (Aïcha)-[:PARTICIPATED_IN]->(Battle of Uhud)
   - Propriétés: {{role: "Medical support", date: "3 AH"}}

8. WITNESSED (شهد)
   - (Sahabi)-[:WITNESSED]->(Treaty of Hudaybiya)
   - Différence avec PARTICIPATED: Témoin passif vs acteur

9. ORGANIZED / LED (نظم / قاد)
   - (Prophète)-[:LED]->(Battle of Badr)
   - (Khalid ibn Walid)-[:LED]->(Saraya expedition)

10. DIED_IN (توفي في)
    - (Hamza)-[:DIED_IN]->(Battle of Uhud)
    - Propriétés: {{manner: "martyred", date: "3 AH"}}

**SPATIALES (مكانية)**

11. LIVED_IN (سكن في)
    - (Prophète)-[:LIVED_IN]->(Mecca)
    - Propriétés: {{period: "Before Hijra", duration_years: 40}}

12. BORN_IN (ولد في)
    - (Prophète)-[:BORN_IN]->(Mecca)
    - Propriétés: {{date: "570 CE", location_detail: "Year of Elephant"}}

13. TRAVELED_TO (سافر إلى)
    - (Prophète)-[:TRAVELED_TO]->(Syria)
    - Propriétés: {{purpose: "Trade", age: 25}}

14. MIGRATED_TO (هاجر إلى)
    - (Muhajirun)-[:MIGRATED_TO]->(Medina)
    - Propriétés: {{event: "Hijra", year: "622 CE / 1 AH"}}

**TEMPORELLES (زمنية)**

15. OCCURRED_DURING (وقع في)
    - (Battle of Badr)-[:OCCURRED_DURING]->(Medinan Period)
    - Propriétés: {{date_hijri: "2 AH", season: "Ramadan"}}

16. BEFORE / AFTER (قبل / بعد)
    - (Battle of Uhud)-[:AFTER]->(Battle of Badr)
    - Propriétés: {{time_gap: "1 year"}}

**SOCIALES (اجتماعية)**

17. TAUGHT (علّم)
    - (Aïcha)-[:TAUGHT]->(Sahaba scholars)
    - Propriétés: {{domain: "Hadith", student_count: "numerous"}}

18. TRANSMITTED_HADITH_FROM (روى عن)
    - (Aïcha)-[:TRANSMITTED_HADITH_FROM]->(Prophète)
    - Propriétés: {{hadith_count: 2210, authenticity: "Sahih"}}

19. PROTECTED / DEFENDED (حمى / دافع عن)
    - (Hamza)-[:DEFENDED]->(Prophète)
    - Contexte: Battle of Uhud

RÈGLES D'EXTRACTION CRITIQUES:

1. **Relations explicites uniquement**
   - ✅ "Aïcha, fille d'Abu Bakr" → DAUGHTER_OF
   - ❌ "Aïcha était sage" → PAS de relation (attribut personnel)

2. **Direction des relations**
   - Familiales: Enfant → Parent (DAUGHTER_OF, SON_OF)
   - Mariage: Bidirectionnel MARRIED_TO (pas de direction)
   - Événements: Personne → Événement (PARTICIPATED_IN)

3. **Propriétés enrichies**
   - Date (Hijri si possible): "3 AH", "Ramadan 2 AH"
   - Lieu si pertinent: "Medina", "Mecca"
   - Rôle si applicable: "commander", "medic", "witness"

4. **Éviter inférences spéculatives**
   - ✅ Texte dit "Aïcha a soigné les blessés à Uhud" → PARTICIPATED_IN
   - ❌ Texte dit "Aïcha était jeune" → PAS de relation AGE (pas un triplet)

5. **Relations multiples OK**
   - (Aïcha)-[:DAUGHTER_OF]->(Abu Bakr)
   - (Aïcha)-[:MARRIED_TO]->(Prophète)
   - (Aïcha)-[:MEMBER_OF_TRIBE]->(Quraysh)
   - Toutes valides simultanément

FORMAT SORTIE JSON:

{{
    "relations": [
        {{
            "source_entity": "Nom exact entité source",
            "relation_type": "TYPE_RELATION",
            "target_entity": "Nom exact entité cible",
            "properties": {{
                "date": "3 AH",
                "location": "Medina",
                "role": "participant"
            }},
            "context_evidence": "Citation exacte du texte justifiant cette relation",
            "confidence": 0.0-1.0
        }}
    ]
}}

EXEMPLES CONCRETS:

Texte: "Aïcha bint Abi Bakr, de la tribu Quraysh, a épousé le Prophète (ﷺ) à Médine. Elle a participé à Uhud en soignant les blessés."

Extraction:
{{
    "relations": [
        {{
            "source_entity": "Aïcha bint Abi Bakr",
            "relation_type": "DAUGHTER_OF",
            "target_entity": "Abu Bakr al-Siddiq",
            "properties": {{}},
            "context_evidence": "Aïcha bint Abi Bakr (bint = fille de)",
            "confidence": 1.0
        }},
        {{
            "source_entity": "Aïcha bint Abi Bakr",
            "relation_type": "MEMBER_OF_TRIBE",
            "target_entity": "Quraysh",
            "properties": {{}},
            "context_evidence": "de la tribu Quraysh",
            "confidence": 1.0
        }},
        {{
            "source_entity": "Aïcha bint Abi Bakr",
            "relation_type": "MARRIED_TO",
            "target_entity": "Prophet Muhammad",
            "properties": {{"location": "Medina"}},
            "context_evidence": "a épousé le Prophète (ﷺ) à Médine",
            "confidence": 1.0
        }},
        {{
            "source_entity": "Aïcha bint Abi Bakr",
            "relation_type": "PARTICIPATED_IN",
            "target_entity": "Battle of Uhud",
            "properties": {{"role": "Medical support"}},
            "context_evidence": "a participé à Uhud en soignant les blessés",
            "confidence": 1.0
        }}
    ]
}}

COMMENCE L'EXTRACTION (JSON uniquement):
"""

# ============================================================

SIRA_RELATION_PASS2_GLEANING = """
Tu es un expert en REVIEW d'extraction de relations Sira.

CONTEXTE DU DOCUMENT:
{identity_context}

CHUNK ANALYSÉ:
{chunk_text}

ENTITÉS VALIDÉES:
{final_entities}

RELATIONS DÉJÀ EXTRAITES (Pass 1):
{relations_pass1}

MISSION: Trouve les relations RATÉES lors du Pass 1.

RELATIONS SOUVENT RATÉES:

1. **Relations implicites via filiation**
   - Texte: "La fille du Prophète a épousé Ali"
   - Pass 1 a peut-être raté: (Ali)-[:MARRIED_TO]->(Fatima)
   - OU raté: (Fatima)-[:DAUGHTER_OF]->(Prophète)

2. **Relations via pronoms**
   - "Aïcha et sa sœur Asma" → (Aïcha)-[:SIBLING]->(Asma)

3. **Relations tribales indirectes**
   - "al-Qurashi" mentionné → MEMBER_OF_TRIBE Quraysh

4. **Relations temporelles**
   - "Après Badr, Uhud a eu lieu" → (Uhud)-[:AFTER]->(Badr)

5. **Relations enseignement/transmission**
   - "Aïcha a transmis des hadiths" → TRANSMITTED_HADITH_FROM Prophète

RÈGLES GLEANING:

- ✅ Ajoute SI relation explicite mais ratée
- ❌ N'ajoute PAS si spéculatif
- ✅ Vérifie relations bidirectionnelles (MARRIED_TO)
- ❌ Évite doublons (vérifier relations_pass1)

FORMAT SORTIE JSON:

{{
    "additional_relations": [
        {{
            "source_entity": "Entity A",
            "relation_type": "TYPE",
            "target_entity": "Entity B",
            "properties": {{}},
            "context_evidence": "Pourquoi ratée en Pass 1",
            "confidence": 0.0-1.0
        }}
    ],
    "rejected_candidates": [
        {{
            "relation": "(A)-[TYPE]->(B)",
            "reason": "Pourquoi rejeté (doublon/spéculatif)"
        }}
    ]
}}

Si AUCUNE relation ratée:
{{
    "additional_relations": [],
    "rejected_candidates": []
}}

COMMENCE LA REVIEW (JSON uniquement):
"""

# ============================================================
# STRUCTURE MODULAIRE POUR FUTURS DOMAINES
# ============================================================

PROMPTS_REGISTRY: Dict[str, Dict[str, str]] = {
    "sira": {
        "entity_pass1": SIRA_ENTITY_PASS1,
        "entity_pass2": SIRA_ENTITY_PASS2_GLEANING,
        "relation_pass1": SIRA_RELATION_PASS1,
        "relation_pass2": SIRA_RELATION_PASS2_GLEANING,
    },
    # Future domaines:
    # "fiqh": {
    #     "entity_pass1": FIQH_ENTITY_PASS1,
    #     ...
    # },
    # "hadith": {...},
}

def get_prompts(domain: str = "sira") -> Dict[str, str]:
    """
    Récupère les prompts pour un domaine spécifique
    
    Args:
        domain: "sira" | "fiqh" | "hadith"
    
    Returns:
        Dict avec entity_pass1, entity_pass2, relation_pass1, relation_pass2
    """
    if domain not in PROMPTS_REGISTRY:
        raise ValueError(f"Domain '{domain}' not found. Available: {list(PROMPTS_REGISTRY.keys())}")
    
    return PROMPTS_REGISTRY[domain]