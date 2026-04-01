# AGENT-ENUM

LLM system prompt enumeration and pipeline detection tool. Analyzes probe/response pairs to reconstruct hidden system prompts, detect multi-component LLM pipeline architectures, and suggest red team attack strategies.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp .env.example .env

# Run (requires Ollama or LM Studio running locally)
python main.py
```

Open `http://127.0.0.1:8765` in your browser.

## Backends

**Ollama** (default): Install from [ollama.ai](https://ollama.ai), then `ollama pull llama3`.

**LM Studio**: Install from [lmstudio.ai](https://lmstudio.ai), load a model, and enable the local server.

Switch backends in the Connection panel at the top of the UI — no restart required.

## How it works

1. Paste a probe you sent to a target LLM and its response
2. Click ANALYZE — the LangGraph pipeline runs:
   - **ResponseAnalyzer** classifies the response (leak/refusal/tool disclosure/neutral)
   - **FragmentExtractor** pulls leaked prompt text on leaks
   - **RefusalClassifier** categorizes refusal types and trigger phrases
   - **RefusalBisector** binary-searches to isolate the minimal trigger phrase
   - **PipelineDetector** infers the target's multi-component architecture
   - **PromptReconstructor** assembles discovered fragments into the reconstructed prompt
   - **StrategyAdvisor** suggests next attack probes
3. Results update live via WebSocket

## API

- `POST /sessions` — create session
- `PATCH /sessions/{id}` — rename session
- `POST /sessions/{id}/probe` — submit probe+response for analysis
- `GET /sessions/{id}` — full session state
- `WS /ws/{id}` — real-time event stream
- `GET /llm/models` — list available models
- `POST /llm/config` — switch backend/model at runtime
