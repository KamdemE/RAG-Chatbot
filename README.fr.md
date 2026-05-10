# Chatbot RAG pour Documentation Technique

Un chatbot basé sur la Génération Augmentée par Récupération (RAG) qui répond aux questions sur un manuel de voiture MG ZS, en utilisant LangChain, Google Gemini et ChromaDB. Le pipeline implémente le **RAG Fusion** avec la méthode de **Reciprocal Rank Fusion (RRF)** pour améliorer la qualité de la récupération.

---

## Concepts Clés

### RAG (Retrieval-Augmented Generation)
Plutôt que de se fier aux données d'entraînement du LLM, le RAG récupère les passages pertinents d'un document au moment de la requête et les injecte dans le prompt. Cela ancre la réponse du modèle dans la source et prévient les hallucinations.

```
Document → Découpage → Embedding → Vector Store
                                        ↓
Question → Embedding → Recherche → Contexte → LLM → Réponse
```

### RAG Fusion + Reciprocal Rank Fusion (RRF)
Une seule requête peut ne pas correspondre à tous les passages pertinents. Le RAG Fusion génère **N variantes** de la question, récupère des documents pour chacune, puis fusionne les listes classées avec RRF.

Formule de score RRF :
```
score(doc) = Σ  1 / (rang_i + k)
```
où `rang_i` est la position du document dans la liste `i` et `k=60` est une constante de lissage.

**Complexité temporelle :** O(N × M × log(N × M)) — N listes, M documents par liste.

---

## Structure des Fichiers

```
rag-chatbot/
├── FirstRAGproject.ipynb   # Notebook principal — pipeline complet commenté
├── data/
│   └── mg-zs-warning-messages.html  
├── README.md
├── README.fr.md
└── LICENSE
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/[github.com/VOTRE_PSEUDO]/RAG-Chatbot.git
cd rag-chatbot-mg-zs

# 2. Installer les dépendances
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
pip install google_genai langchain-google-genai langchain-chroma langchain-text-splitters
```

---

## Configuration

Avant d'exécuter, ouvrez `FirstRAGproject.ipynb` et remplacez les valeurs dans la cellule 5 :

| Variable | Description |
|---|---|
| `API` | Votre clé API Google Gemini — gratuite sur [aistudio.google.com](https://aistudio.google.com) |
| `LANGCHAIN_API_KEY` | Votre clé LangSmith (optionnelle) — gratuite sur [smith.langchain.com](https://smith.langchain.com) |
| `sources` | Chemin absolu vers `mg-zs-warning-messages.html` sur votre machine |

---

## Exécution

```bash
jupyter notebook FirstRAGproject.ipynb
```

Exécutez toutes les cellules dans l'ordre. La dernière cellule invoque la chaîne RAG et affiche la réponse.

---

## Exemple de Résultat

Document source : `mg-zs-warning-messages.html` (manuel MG ZS — section messages d'avertissement)

Requête :
```
"What are the warnings to care about"
```

Résultat :
```
Voici les avertissements à prendre en compte :

• Stop Start System Fault — système Stop/Start défaillant. Consulter un réparateur agréé MG dès que possible.
• Gasoline Particular Filter Full — filtre à particules plein. Consulter un réparateur agréé MG dès que possible.
• Engine Coolant Temperature High — température élevée du liquide de refroidissement. Arrêter le véhicule immédiatement.
• ABS Fault — système ABS défaillant. Consulter un réparateur agréé MG immédiatement.
... (et d'autres)
```

---

## Auteurs

- **KAMDEM KOUAM Ezechiel** 
— [github.com/KamdemE](https://github.com/KamdemE)

---

## Licence

MIT — voir [LICENSE](LICENSE)
