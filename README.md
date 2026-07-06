# TecnoBot — Assistant IA basé sur RAG pour l'Éducation en Génie Électrique

Projet de fin d'année — École Nationale d'Ingénieurs de Gabès (ENIG)
Département Génie Électrique et Automatique — Année universitaire 2025-2026

**Auteurs :** Ilyes Harrathi, Malek Chihi, Farah Akkeri
**Encadrants :** Prof. Zaied Mourad, Mme Rania Bengarssalah

📄 Rapport complet du projet : [`TecnoBot_Report_Final.pdf`](./TecnoBot_Report_Final.pdf)

## Description

TecnoBot est un assistant IA spécialisé qui utilise une architecture RAG
(Retrieval-Augmented Generation) pour répondre aux questions techniques
en génie électrique, en s'appuyant sur une base de connaissances locale
(manuels, guides de maintenance, documents techniques) plutôt que sur les
seules connaissances pré-entraînées du modèle. Le système utilise le
modèle **Qwen2.5** exécuté localement via **Ollama**, un pipeline de
récupération sémantique avec **ChromaDB**, et une interface **Streamlit**.

Fonctionnalités principales :
- Chatbot RAG pour questions techniques en génie électrique
- Laboratoire de simulation de signaux (AC, DC, triphasé, loi d'Ohm, PWM)
- Tableau de bord système et gestion des utilisateurs par rôle

## Architecture

```
PDF Documents → Chunking → Embedding → ChromaDB (Vector Store)
User Query → Query Embedding → Semantic Search → Local LLM (Qwen2.5) → Réponse
```

## Évolution du projet

Le projet a été développé de manière itérative, en deux phases majeures
documentées en détail dans le rapport :

| Phase | Modèle LLM | Points clés |
|---|---|---|
| **Phase 1** — Prototype | Phi-3 (Microsoft) | RAG basique, ChromaDB, interface Streamlit, authentification simple |
| **Phase 2** — Version actuelle | Qwen2.5 (Alibaba Cloud) | RAG avancé multi-documents, Signal Lab interactif, tableau de bord système, contrôle d'accès par rôle |

La migration de Phi-3 vers Qwen2.5 a permis une meilleure précision
technique, une réduction des hallucinations et une meilleure gestion du
français/arabe. Le dépôt contient la **version finale (Phase 2)** ;
le détail complet des deux phases, avec captures d'écran et résultats
expérimentaux, est disponible dans le rapport PDF ci-dessus.

## Structure du projet

```
projetai/
├── app.py               # Application principale (Streamlit + RAG + Qwen2.5)
├── requirements.txt     # Dépendances Python
├── README.md
├── TecnoBot_Report_Final.pdf   # Rapport détaillé (Phase 1 & 2, résultats, tests)
├── src/
│   └── pdf_loader.py    # Script de chargement et découpage des PDF
└── data/                 # Documents techniques (PDF) constituant la base de connaissances
    └── *.pdf
```

> Le dossier `chroma_db/` (base vectorielle) est généré localement au
> premier lancement et n'est pas versionné (voir `.gitignore`). Il se
> reconstruit automatiquement à partir des PDF du dossier `data/`.

## Installation

```bash
# 1. Cloner le repo
git clone <URL_DU_REPO>
cd projetai

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Installer Ollama et récupérer les modèles nécessaires
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 5. Lancer l'application
streamlit run app.py
```

## Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| LLM local | Qwen2.5:7b via Ollama |
| Embeddings | nomic-embed-text |
| Base vectorielle | ChromaDB |
| Orchestration RAG | LangChain |

## Licence

Projet académique — usage éducatif, ENIG 2025-2026.
