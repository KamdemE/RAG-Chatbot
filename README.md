# RAG Chatbot for Technical Documentation

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about an MG ZS car manual using LangChain, Google Gemini, and ChromaDB. The pipeline implements **RAG Fusion** with **Reciprocal Rank Fusion (RRF)** for improved retrieval quality.

---

## Key Concepts

### RAG (Retrieval-Augmented Generation)
Rather than relying on the LLM's training data, RAG retrieves relevant passages from a document at query time and injects them into the prompt. This grounds the model's answer in the source material and prevents hallucinations.

```
Document → Chunk → Embed → Vector Store
                                ↓
Question → Embed → Similarity Search → Context → LLM → Answer
```

### RAG Fusion + Reciprocal Rank Fusion (RRF)
A single query may not match all relevant chunks. RAG Fusion generates **N paraphrased versions** of the question, retrieves documents for each, then merges the ranked lists using RRF.

RRF score formula:
```
score(doc) = Σ  1 / (rank_i + k)
```
where `rank_i` is the document's position in list `i` and `k=60` is a smoothing constant.

**Time complexity:** O(N × M × log(N × M)) — N lists, M documents per list.

---

## File Structure

```
rag-chatbot/
├── FirstRAGproject.ipynb   # Main notebook — full commented pipeline
├── data/
│   └── mg-zs-warning-messages.html  # ⚠️ Not included — see Data section below
├── README.md
├── README.fr.md
└── LICENSE
```

---

## Data

> ⚠️ **The HTML file `mg-zs-warning-messages.html` is NOT included in this repository.**
>
> This file is extracted from the MG ZS owner's manual and may be subject to MG Motor's copyright. Please obtain it from your own vehicle documentation or a licensed source. Once obtained, place it in the `data/` folder and update the `sources` variable in the notebook accordingly.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/[github.com/YOUR_USERNAME]/rag-chatbot-mg-zs.git
cd rag-chatbot-mg-zs

# 2. Install dependencies
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
pip install google_genai langchain-google-genai langchain-chroma langchain-text-splitters
```

---

## Configuration

Before running, open `FirstRAGproject.ipynb` and replace the placeholders in cell 5:

| Variable | Description |
|---|---|
| `API` | Your Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com) |
| `LANGCHAIN_API_KEY` | Your LangSmith key (optional) — free at [smith.langchain.com](https://smith.langchain.com) |
| `sources` | Absolute path to `mg-zs-warning-messages.html` on your machine |

---

## Execution

```bash
jupyter notebook FirstRAGproject.ipynb
```

Run all cells in order. The final cell invokes the RAG chain and prints the answer.

---

## Example Output

Input document: `mg-zs-warning-messages.html` (MG ZS car manual — warning messages section)

Query:
```
"What are the warnings to care about"
```

Output:
```
Voici les avertissements à prendre en compte :

• Stop Start System Fault — système Stop/Start défaillant. Consulter un réparateur agréé MG dès que possible.
• Gasoline Particular Filter Full — filtre à particules plein. Consulter un réparateur agréé MG dès que possible.
• Engine Coolant Temperature High — température élevée du liquide de refroidissement. Arrêter le véhicule immédiatement.
• ABS Fault — système ABS défaillant. Consulter un réparateur agréé MG immédiatement.
... (and more)
```

---

## Authors

- **[YOUR NAME]** — [github.com/YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## License

MIT — see [LICENSE](LICENSE)
