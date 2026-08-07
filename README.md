# SafeBirth Prototype

A working pipeline for the hackathon build: patient registration, emergency call
transcription (Whisper), AI risk classification, and a healthcare worker dashboard,
all in one Gradio app.

## How it maps to the proposal

| Proposal module | Code |
|---|---|
| Patient Registration Module | `register_patient()`, name + community identification with automatic numbering for duplicate names |
| Emergency Voice Call Simulation | Audio upload field in the "Emergency Call" tab |
| AI Speech to Text | `transcribe_audio()` using Hugging Face Whisper |
| AI Maternal Risk Analysis | `classify_risk()`, uses Claude if an API key is set, otherwise a transparent rule-based fallback |
| Healthcare Worker Dashboard | "Healthcare Worker Dashboard" tab, live alert table |

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

This opens a local web UI (usually at `http://127.0.0.1:7860`).

## Run it on Google Colab (recommended if you don't have a GPU)

1. Upload `app.py` to a new Colab notebook's file browser
2. In a cell, run:
   ```
   !pip install -r requirements.txt
   ```
3. In another cell:
   ```python
   %run app.py
   ```
4. Colab will give you a public `gradio.live` link you can use for the demo

## Deploy to a Hugging Face Space (recommended for the actual hackathon demo)

1. Go to huggingface.co, create a new **Space**, choose the **Gradio** SDK
2. Upload `app.py` and `requirements.txt` to the Space
3. (Optional) In the Space's **Settings > Repository secrets**, add `ANTHROPIC_API_KEY`
   if you want Claude-based classification instead of the rule-based fallback
4. The Space builds automatically and gives you a public URL, this is what you'd
   show judges

## Using it without an API key

The app works fully without any API key, `classify_risk()` falls back to a transparent,
guideline-derived rule-based classifier (checks for critical terms like "unconscious" or
"heavy bleeding", and cross-references high-risk terms against the patient's known
conditions like hypertension). This is actually a legitimate design choice to mention in
your pitch: traceable logic instead of a black box, in a context where lives are at stake.

If you do want Claude-based classification (more nuanced reasoning over the transcript
and history), set the environment variable before running:

```bash
export ANTHROPIC_API_KEY=your_key_here
python app.py
```

## Testing without real audio

The "Emergency Call" tab has a manual transcript text box, type a symptom description
directly (e.g. "she is bleeding heavily and feels dizzy") to test the classification
pipeline without needing to record or upload audio first.

## Known limitations to mention in the demo

- Whisper's Twi support is limited since it wasn't a major part of Whisper's training
  data, test your actual Twi audio clips early and have a typed-transcript fallback
  ready for the live demo if transcription quality is poor.
- The in-memory patient store resets every time the app restarts, fine for a demo,
  swap in Firebase/Supabase/Airtable for anything persistent.
- The rule-based classifier is intentionally simple, good enough to demonstrate the
  concept, not a substitute for a real clinical model.
