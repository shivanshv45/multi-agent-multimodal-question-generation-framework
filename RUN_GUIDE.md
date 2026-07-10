# 🏃‍♂️ Tri-Agent Swarm — Official Run Guide

This guide explains exactly how to run the multi-agent pipeline to generate culturally-grounded, code-mixed multimodal benchmarks for your research project.

---

## 1. Where to Put Your Files

You need to organize your source media (images, videos, audio) into their respective language folders before running the batch commands. 

**Example Folder Structure:**
```text
trichy/
│
├── dataset_tamil/            <-- Put your Tamil/South Indian images here
│   ├── 1.png
│   └── kolam.jpg
│
├── dataset_hindi/            <-- Put your Hindi/North Indian images here
│
├── dataset_telugu/           <-- Put your Telugu images here
│
├── output/                   <-- The pipeline will automatically create subfolders here
│
├── triagent/                 <-- The core codebase
└── ...
```

---

## 2. API Key Setup
Before running the commands, ensure your Gemini API key is active. The pipeline uses Google Gemini by default because of its native multimodal capabilities (though you can change the Reasoning Agent to Grok/Ollama in `config.py`).

You must have an environment variable set, or a `.env` file in the root of the project with:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

*(You can verify everything is connected by running `python -m triagent check`)*

---

## 3. The Commands (Dual-Language Simultaneous Generation)

**🔥 MASSIVE UPGRADE:** The framework no longer requires you to run separate commands for Tanglish and Tamil. 
By running a single command for a "Base Language", the Synthesis Agent will automatically pack **THREE** distinct formats into a single JSON file simultaneously:
1. **Pure Regional Format** (e.g., proper Tamil script)
2. **Code-Mixed Format** (e.g., Tanglish)
3. **English Baseline** (For pure logic comparison)

Open your terminal (PowerShell or Command Prompt) in the `trichy` folder and run these commands one by one:

### ▶️ 1. Generate the Tamil/Tanglish/English Dataset
```powershell
$env:PYTHONUTF8=1; python -m triagent batch --media-dir "./dataset_tamil" -l "Tamil" -o "./output/tamil"
```

### ▶️ 2. Generate the Hindi/Hinglish/English Dataset
```powershell
$env:PYTHONUTF8=1; python -m triagent batch --media-dir "./dataset_hindi" -l "Hindi" -o "./output/hindi"
```

### ▶️ 3. Generate the Telugu/Teluguish/English Dataset
```powershell
$env:PYTHONUTF8=1; python -m triagent batch --media-dir "./dataset_telugu" -l "Telugu" -o "./output/telugu"
```

*(Note: The `$env:PYTHONUTF8=1` ensures Windows PowerShell doesn't crash when printing the beautiful UI borders).*

---

## 4. Where is the Output?

If you run the commands above, the pipeline will automatically create organized folders inside `output/`:
* `output/tamil/`
* `output/hindi/`
* `output/telugu/`

Inside these folders, you will find perfectly formatted `.json` files. The filename will always include the original image name so you can easily reference it! (e.g., `1_taq-84bd7a.json`).

Each JSON contains the full traceable thought process:
1. `visual_ir`: The structural extraction (Phase 1)
2. `reasoning_output`: The logical skeleton and distractors (Phase 2)
3. `question_stem_pure`, `question_stem_mixed`, `question_stem_english`: The final synthesized benchmark item (Phase 3)

---

## 5. Single File Quick Test (Demo)

If you just want to test a single image to see the agents "think" in real-time before running a massive batch folder, use the run command:

```powershell
$env:PYTHONUTF8=1; python -m triagent run --media "./dataset_tamil/1.png" -l "Tamil" -o "./output/tamil"
```
