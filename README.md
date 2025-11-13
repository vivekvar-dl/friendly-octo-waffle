# Legal RAG Backend Prototype

Prototype FastAPI backend that augments `Qwen/Qwen3-8B` with legal PDFs (POCSO & BNSS Acts) via [LlamaIndex](https://www.llamaindex.ai/).

## Prerequisites

- Python 3.10+ (tested with 3.13.9)
- GPU with ~16 GB VRAM for full-precision inference, or enable 4-bit quantization (default) with a compatible CUDA setup (`bitsandbytes`)
- (Optional) `uvicorn`/`httpie` for local testing

> **Note:** Download the POCSO and BNSS PDF files and place them in `data/pdfs/` before ingestion.

## Setup

```powershell
cd "E:\AI4AP police\testllm"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need to adjust model paths or disable 4-bit loading, create an `.env` file:

```
MODEL_NAME=Qwen/Qwen3-8B
LOAD_IN_4BIT=false
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
```

## Running the API

```powershell
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`. Open `/docs` for the interactive Swagger UI.

## Container Image

Build the Docker image (from the project root):

```powershell
docker build -t legal-rag-backend .
```

Run the container, mounting your PDFs and persisted index so they survive rebuilds:

```powershell
docker run --gpus all `
    -p 8000:8000 `
    -v ${PWD}\data\pdfs:/app/data/pdfs `
    -v ${PWD}\storage\index:/app/storage/index `
    --env-file .env `
    legal-rag-backend
```

> **GPU prerequisites:** install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your Azure VM and ensure the VM image already has the matching CUDA driver. The container ships with CUDA 12.1 runtime; confirm your host driver supports it (driver ≥ 535).

The service is now available at `http://localhost:8000`. Use the same ingestion and query endpoints as described earlier.

## Workflow

1. **Ingest PDFs** (builds or rebuilds the vector index)

   ```powershell
   curl -X POST http://localhost:8000/api/ingest
   ```

2. **Query** (retrieves relevant passages and synthesizes an answer)

   ```powershell
   curl -X POST http://localhost:8000/api/query `
        -H "Content-Type: application/json" `
        -d '{ "question": "What are the reporting obligations under the POCSO Act?" }'
   ```

The response includes the answer plus retrieved source chunks (`file_name`, `page_number`, and `text`).

## Notes & Next Steps

- The first query initializes the Qwen model; expect a longer load time.
- Persisted indices live in `storage/index/`; delete this directory to force a full rebuild.
- Consider adding background jobs or task queues for ingestion if PDF updates become frequent.
- Add authentication, request logging, and evaluation harnesses before promoting beyond prototype.

