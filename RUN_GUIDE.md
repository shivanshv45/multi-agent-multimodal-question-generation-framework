# 🏃‍♂️ Tri-Agent Swarm — Official Run Guide

This guide explains exactly how to run the multi-agent pipeline to generate culturally-grounded, code-mixed multimodal benchmarks for your research project.

---

## 1. Where to Put Your Files

You need to organize your source media (images, videos, audio) before running the batch commands. 

Create a folder in your project root called `dataset`. Inside it, you can drop any supported media file:
* **Images:** `.jpg`, `.png`, `.webp`
* **Videos:** `.mp4`, `.mov`, `.avi`
* **Audio:** `.mp3`, `.wav`, `.m4a`

**Example Folder Structure:**
```text
trichy/
│
├── dataset/                  <-- Put all your source files here!
│   ├── kolam.jpg
│   ├── temple_festival.mp4
│   └── traditional_song.mp3
│
├── output/                   <-- The pipeline will automatically create this and save the JSONs here
│
├── triagent/                 <-- The core codebase
└── ...
```

---

## 2. API Key Setup
Before running the commands, ensure your Gemini API key is active. The pipeline uses Google Gemini by default because of its native multimodal capabilities.

You must have an environment variable set, or a `.env` file in the root of the project with:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

*(You can verify everything is connected by running `python -m triagent check`)*

---

## 3. The Commands to Generate All Languages

To generate the highest-quality grammar and cultural tone, we process one language per command run. Since you want to generate benchmarks for **Tamil, Tanglish, Telugu, Teluguish, Hindi, and Hinglish**, you will simply run the `batch` command pointing to your `dataset/` folder for each language.

Open your terminal (PowerShell or Command Prompt) in the `trichy` folder and run these commands one by one:

### ▶️ 1. Tanglish (Tamil-English)
```powershell
$env:PYTHONUTF8=1
python -m triagent batch --media-dir ./dataset -l "Tanglish (Tamil-English)" -o ./output/tanglish
```

### ▶️ 2. Pure Tamil
```powershell
$env:PYTHONUTF8=1
python -m triagent batch --media-dir ./dataset -l "Tamil" -o ./output/tamil
```

### ▶️ 3. Hinglish (Hindi-English)
```powershell
$env:PYTHONUTF8=1
python -m triagent batch --media-dir ./dataset -l "Hinglish (Hindi-English)" -o ./output/hinglish
```

### ▶️ 4. Pure Hindi
```powershell
$env:PYTHONUTF8=1
python -m triagent batch --media-dir ./dataset -l "Hindi" -o ./output/hindi
```

### ▶️ 5. Teluguish (Telugu-English)
```powershell
$env:PYTHONUTF8=1
python -m triagent batch --media-dir ./dataset -l "Teluguish (Telugu-English)" -o ./output/teluguish
```

### ▶️ 6. Pure Telugu
```powershell
$env:PYTHONUTF8=1
python -m triagent batch --media-dir ./dataset -l "Telugu" -o ./output/telugu
```

*(Note: The `$env:PYTHONUTF8=1` ensures Windows PowerShell doesn't crash when trying to print beautiful UI borders to the terminal).*

---

## 4. Where is the Output?

If you run the commands above, the pipeline will automatically create organized folders inside `output/`:
* `output/tanglish/`
* `output/hinglish/`
* etc.

Inside these folders, you will find perfectly formatted `.json` files. Each JSON contains the full traceable thought process:
1. `visual_ir` / `video_ir`: The structural extraction (Phase 1)
2. `reasoning_output`: The logical skeleton and distractors (Phase 2)
3. `question_stem`, `choices`, `correct_answer`: The final synthesized benchmark item (Phase 3)

---

## 5. Single File Quick Test (Demo)

If you just want to test a single image/video to see the agents "think" in real-time before running a massive batch folder, use the run command:

```powershell
$env:PYTHONUTF8=1
python -m triagent run --media ./dataset/kolam.jpg -l "Tanglish (Tamil-English)"
```
