# 🧠 Tri-Agent Swarm: Multi-Agent Multimodal Question Generation Framework

**Institution:** National Institute of Technology, Tiruchirappalli  
**Department:** Computer Science and Engineering  
**Supervisor:** Dr. A. Santhanavijayan  
**Research Focus:** Culturally Grounded Multimodal Reasoning & Code-Mixed NLP

---


### The Three Agents

1. **Visual Context Agent (The Grounder)** — Parses images into structured Intermediate Representations (IR) with cultural markers, spatial relations, and semantic primitives.

2. **Reasoning Agent (The Logician)** — Operates *blindly* on the IR (never sees the image). Performs cognitive routing and structural analogy mapping to generate logical question structures.

3. **Synthesis Agent (The Linguist)** — Generates code-mixed (Tamil-English) MCQ benchmark items with ground-truth answers and culturally-grounded distractors.

## Supported Model Backends

| Backend | Models | Type |
|---------|--------|------|
| **Google Gemini** | gemini-2.5-flash, gemini-2.5-pro | Cloud API |
| **xAI Grok** | grok-3, grok-3-mini | Cloud API |
| **Ollama** | llama3, mistral, llava, etc. | Local | in progess , 

| will add more models later |

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
copy .env.example .env
# Edit .env with your API keys

# 4. Run the demo pipeline
python -m triagent.demo

# 5. Run with a specific image
python -m triagent.cli --image path/to/image.jpg --backend gemini
```

## Project Structure

```
trichy/
├── triagent/
│   ├── __init__.py
│   ├── config.py          # Configuration & env management
│   ├── schemas.py         # IR JSON schema & Pydantic models
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py        # Abstract backend interface
│   │   ├── gemini.py      # Google Gemini API client
│   │   ├── grok.py        # xAI Grok API client
│   │   └── ollama.py      # Ollama local inference client
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py        # Abstract agent interface
│   │   ├── visual.py      # Phase 1: Visual Context Agent
│   │   ├── reasoning.py   # Phase 2: Reasoning Agent
│   │   └── synthesis.py   # Phase 3: Synthesis Agent
│   ├── pipeline.py        # Orchestrator / Swarm Controller
│   ├── cli.py             # CLI entry point
│   └── demo.py            # Demo with sample images
├── data/
│   └── sample_images/     # Sample test images
├── output/                # Generated benchmark items
├── requirements.txt
├── .env.example
└── README.md
```

## License

Research use only — NIT Trichy CSE Department
