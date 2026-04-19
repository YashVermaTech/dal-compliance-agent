# ✈️ DAL-Aware Compliance Agent

I built this because DO-178C certification is genuinely painful.

If you've ever spent hours searching through a 500-page standard document trying to figure out whether your DAL-B software needs decision coverage or MC/DC — this tool is for you.

## What it does

You upload your compliance documents. You set your DAL level. You ask questions. It answers with exact section and page references, filtered to your specific assurance level so you're not getting DAL-A answers when you're working on DAL-C.

Five things it can do:

**1. Compliance Chat** — ask anything about DO-178C, DO-254, ARP4761, ARP4754A or DO-160 in English or German. Every answer cites the exact section and page it came from.

**2. Traceability Matrix** — paste your requirements and get a full matrix linking each one to verification methods, test cases and coverage status.

**3. Gap Analysis** — upload your project documents and find out exactly which DO-178C objectives you're missing before your DER does.

**4. Impact Analyzer** — describe a software change and get a risk assessment showing which requirements, tests and documents are affected.

**5. Document Ingestion** — upload PDFs directly. The system chunks them intelligently by section structure and indexes them with DAL-level metadata.

## Why I built it this way

The hardest part of compliance work isn't understanding the standard — it's finding the right paragraph at the right time, and knowing whether it applies to your specific DAL.

Standard RAG systems don't solve this because they retrieve by semantic similarity without any awareness of assurance levels. A DAL-A question should never pull DAL-C content. So I built a DAL-context agent that sets the assurance level first and filters every retrieval through that context.

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | Streamlit on HuggingFace Spaces | Free, fast to build |
| Backend | FastAPI on Render.com | Free tier, auto-deploys from GitHub |
| Vector DB | Pinecone serverless | Free tier, handles DAL metadata filtering |
| LLM | Groq llama3-70b | Free tier, fast enough for real use |
| Agent | LangChain with 6 tools | Structured tool calling with memory |
| Embeddings | Pinecone Inference API | No local model, runs within 512MB RAM |

Everything runs free. No credit card needed to try it.

## Try it

Live app: https://huggingface.co/spaces/YashVer/dal-compliance-agent
API docs: https://dal-compliance-agent.onrender.com/docs

Note: the Render free tier spins down after 15 minutes of inactivity. First request after sleep takes about 50 seconds. After that it's fast.

## Run it yourself

You need two free API keys:

Groq (the LLM): console.groq.com → API Keys → Create key

Pinecone (the vector store): pinecone.io → Create account → Create index named dal-compliance-index, dimensions 1024, metric cosine

Clone the repo and create a .env file:
GROQ_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_ENV=us-east-1
PINECONE_INDEX_NAME=dal-compliance-index
GROQ_MODEL=llama3-70b-8192
BACKEND_URL=http://localhost:8000

Run the backend:
cd dal-compliance-agent
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload

Run the frontend:
pip install -r requirements.txt
streamlit run src/streamlit_app.py

## The 6 agent tools

The LangChain agent has access to these tools:
- set_dal_level — stores your DAL for the session, filters all subsequent retrievals
- query_compliance — semantic search with DAL filtering, returns answer + citations
- generate_traceability_matrix — parses requirements and maps them to DO-178C objectives
- detect_compliance_gaps — runs your project docs against a 71-objective checklist
- dal_impact_analyzer — change impact assessment with risk scoring
- multi_standard_router — detects which standard applies

## DAL reference

| Level | Failure Condition | Objectives | Coverage Required |
|-------|------------------|------------|------------------|
| DAL-A | Catastrophic | 71 | MC/DC |
| DAL-B | Hazardous | 69 | Decision |
| DAL-C | Major | 62 | Statement |
| DAL-D | Minor | 26 | Statement |

## Background

I'm Yash Verma — I wrote my master's thesis at Airbus Aerostructures in Hamburg on deep learning for aircraft fuselage inspection (grade 1.0). Before that I interned at Lilium working on aircraft health monitoring with Palantir Foundry.

This project grew out of frustration with how time-consuming DO-178C compliance work is in practice, and curiosity about whether LLMs could actually help — not just summarize documents but reason about assurance levels and traceability.

M.Sc. Aerospace Engineering, TU Darmstadt
yashverma25104@gmail.com
Portfolio: yashverma-ai.netlify.app

This tool is for engineering assistance only. All certification decisions require review by a qualified Designated Engineering Representative (DER).
