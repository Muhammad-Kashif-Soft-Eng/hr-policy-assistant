# HR Policy Assistant — RAG with Streamlit

A simple Retrieval-Augmented Generation (RAG) application that lets a user upload an HR Policy PDF and ask questions about it.

The app uses:

- **Streamlit** for the web interface
- **PyMuPDF** for PDF text extraction
- **Sentence Transformers** with `all-MiniLM-L6-v2` for semantic embeddings
- **FAISS** for vector similarity search
- **Groq** with `openai/gpt-oss-20b` for grounded answer generation
- **Groq Structured Outputs** with JSON Schema so each answer follows a consistent structure

## How the RAG pipeline works

```text
HR Policy PDF
    ↓
PyMuPDF text extraction
    ↓
Page-aware overlapping chunks
    ↓
Sentence Transformer embeddings
    ↓
FAISS similarity index
    ↓
Top relevant policy passages
    ↓
Groq GPT-OSS 20B
    ↓
Structured response
    ├── Status
    ├── Answer
    ├── Key Points
    ├── Policy Basis
    ├── Caveats
    └── Confidence
```

## Project files

```text
hr-policy-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important: no API key in the repository

Do **not** put the Groq API key inside `app.py`.

The app reads:

```text
GROQ_API_KEY
```

from Streamlit Secrets.

This keeps the secret outside the GitHub repository.

## 1. Get a Groq API key

Open:

https://console.groq.com/keys

Create an API key and keep it private.

Groq currently lists:

```text
openai/gpt-oss-20b
```

as an available model and supports structured JSON-schema output for this model.

## 2. Create the GitHub repository in your browser

You do not need VS Code, terminal, or Colab.

1. Sign in to GitHub.
2. Click **New repository**.
3. Give it a name such as `hr-policy-assistant`.
4. Choose **Public** or **Private**.
5. Create the repository.
6. Open the repository.
7. Click **Add file → Upload files**.
8. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
9. Commit the files to the default branch.

You can upload the four files directly from your browser.

## 3. Connect GitHub to Streamlit Community Cloud

Open:

https://share.streamlit.io/

1. Sign in with GitHub.
2. Connect/authorize your GitHub account when prompted.
3. Click **Create app**.
4. Select **Yup, I have an app**.
5. Select your GitHub repository.
6. Select the branch containing the uploaded files, normally `main`.
7. Set the main file to:

```text
app.py
```

8. Open **Advanced settings**.

## 4. Add the Groq secret

Inside Streamlit's **Advanced settings → Secrets**, enter:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Replace `YOUR_GROQ_API_KEY` with the key you created in Groq.

Do not add the key to GitHub, README.md, or app.py.

## 5. Deploy

Click **Deploy**.

Streamlit Community Cloud will install the packages from `requirements.txt`, start `app.py`, and provide a public `streamlit.app` URL.

The first deployment can take longer because the Sentence Transformer model must be downloaded and loaded.

## 6. Use the deployed app

1. Open your Streamlit URL.
2. Upload one HR Policy PDF.
3. Wait for the searchable index to finish building.
4. Ask an HR policy question.
5. Review the structured response.
6. Check the retrieved PDF pages shown under **Retrieved Sources**.

## Structured answer format

Each successful response is displayed as:

### Answer
A direct response to the question using the retrieved policy text.

### Key Points
The important practical points from the answer.

### Policy Basis
The policy statements from the retrieved context that support the answer.

### Caveats
Missing details, limits, or situations where the policy text is not enough.

### Confidence
`High`, `Medium`, or `Low`.

### Retrieved Sources
The PDF pages and semantic similarity scores for the passages supplied to the model.

## Grounding behavior

The assistant is instructed to:

- use only the retrieved uploaded-policy passages
- avoid inventing HR rules or numbers
- say `NOT_FOUND` when the supplied policy context is insufficient
- show the retrieved source pages
- distinguish missing policy information from an actual answer
- avoid presenting the result as legal advice

## Browser-only workflow

This project is intentionally suitable for a workflow with:

```text
GitHub website
      ↓
Upload files
      ↓
Streamlit Community Cloud
      ↓
Add GROQ_API_KEY in Secrets
      ↓
Deploy
```

No terminal, VS Code, or Google Colab is required.

## Troubleshooting

### `GROQ_API_KEY is missing`

Open the Streamlit app settings and add:

```toml
GROQ_API_KEY = "your-key"
```

under **Secrets**.

### Deployment cannot find dependencies

Make sure `requirements.txt` is in the repository root next to `app.py`.

### The PDF gives no useful text

This implementation works best with text-based PDFs. Scanned image-only PDFs may need OCR before their content can be retrieved.

### The app starts slowly

The first startup may take longer because the embedding model and Python dependencies have to be installed/downloaded.

### GitHub changes are not visible immediately

After committing a new `app.py` or `requirements.txt`, Streamlit Community Cloud normally detects the repository change and updates the app. You can also use the app's **Manage app** controls to inspect deployment logs.

## Security reminder

Treat your Groq API key as a password.

Never commit:

```text
.streamlit/secrets.toml
```

or any file containing the actual API key.
