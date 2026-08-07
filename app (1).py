"""
SafeBirth prototype pipeline
Registration -> Emergency voice call simulation -> Whisper transcription -> AI risk classification -> Healthcare worker dashboard

Run locally:
    pip install -r requirements.txt
    python app.py

Or deploy directly to a Hugging Face Space:
    1. Create a new Space (SDK: Gradio)
    2. Upload app.py and requirements.txt
    3. Add ANTHROPIC_API_KEY as a secret in the Space settings (optional, see note below)
"""

import os
import json
import gradio as gr
from datetime import datetime

# ---------------------------------------------------------------------------
# In-memory "database". Swap this for Firebase/Supabase/Airtable later,
# the rest of the pipeline does not need to change.
# ---------------------------------------------------------------------------
PATIENTS = {}   # key: "name|community" -> patient record dict
ALERTS = []     # list of alert dicts, newest first


def patient_key(name, community):
    return f"{name.strip().lower()}|{community.strip().lower()}"


def register_patient(name, community, age, gestational_age, conditions, language, phone):
    if not name or not community:
        return "Name and community are required.", registration_table()

    key = patient_key(name, community)

    # Handle duplicate names in the same community: append a number,
    # matching the "Ama Mensah 2, Aframso" identification approach in the proposal.
    if key in PATIENTS:
        n = 2
        while patient_key(f"{name} {n}", community) in PATIENTS:
            n += 1
        name = f"{name} {n}"
        key = patient_key(name, community)

    PATIENTS[key] = {
        "name": name,
        "community": community,
        "age": age,
        "gestational_age": gestational_age,
        "conditions": conditions,
        "language": language,
        "phone": phone,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return f"Registered: {name}, {community}", registration_table()


def registration_table():
    rows = [
        [p["name"], p["community"], p["age"], p["gestational_age"], p["conditions"], p["language"]]
        for p in PATIENTS.values()
    ]
    return rows


# ---------------------------------------------------------------------------
# Step 1: Speech to text (Hugging Face Whisper)
# Loaded lazily so the app starts instantly even before the model downloads.
# ---------------------------------------------------------------------------
_whisper_pipe = None

def get_whisper():
    global _whisper_pipe
    if _whisper_pipe is None:
        from transformers import pipeline
        # whisper-small is a good speed/accuracy tradeoff for a live demo.
        # Swap to "openai/whisper-large-v3" for better accuracy if you have GPU time.
        _whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    return _whisper_pipe


def transcribe_audio(audio_path):
    if audio_path is None:
        return ""
    whisper = get_whisper()
    result = whisper(audio_path)
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Step 2: AI risk classification
# Uses the Anthropic API if ANTHROPIC_API_KEY is set, otherwise falls back
# to a transparent rule-based classifier so the demo still works with no key.
# ---------------------------------------------------------------------------
def classify_with_claude(patient, transcript):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    prompt = f"""You are a clinical triage assistant for a maternal health emergency system.

Patient record:
- Name: {patient.get('name')}
- Community: {patient.get('community')}
- Age: {patient.get('age')}
- Gestational age: {patient.get('gestational_age')}
- Known conditions: {patient.get('conditions')}

Emergency call transcript:
"{transcript}"

Classify this case and respond with ONLY valid JSON in this exact shape, no other text:
{{
  "risk_level": "CRITICAL" | "HIGH" | "MODERATE" | "LOW",
  "symptoms": ["list", "of", "extracted", "symptoms"],
  "reason": "one sentence explaining the classification",
  "recommendation": "one sentence recommended next action"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def classify_with_rules(patient, transcript):
    """Transparent fallback classifier, no API key required.
    Mirrors the guideline-derived logic described in the proposal."""
    t = transcript.lower()
    conditions = (patient.get("conditions") or "").lower()

    critical_terms = ["unconscious", "unresponsive", "convulsion", "seizure",
                       "severe bleeding", "heavy bleeding", "can't breathe",
                       "difficulty breathing", "not breathing"]
    high_terms = ["bleeding", "severe headache", "blurred vision", "severe pain",
                  "swollen feet", "swelling"]

    found_critical = [term for term in critical_terms if term in t]
    found_high = [term for term in high_terms if term in t]

    if found_critical:
        risk = "CRITICAL"
        reason = f"Reported symptoms ({', '.join(found_critical)}) indicate an obvious life-threatening emergency."
        rec = "Dispatch emergency response immediately, do not wait for further assessment."
    elif found_high and "hypertension" in conditions:
        risk = "HIGH"
        reason = f"Reported symptoms ({', '.join(found_high)}) combined with a history of hypertension indicate elevated risk."
        rec = "Healthcare worker should review and respond promptly."
    elif found_high:
        risk = "MODERATE"
        reason = f"Reported symptoms ({', '.join(found_high)}) warrant clinical review."
        rec = "Schedule prompt review with a healthcare professional."
    else:
        risk = "LOW"
        reason = "No high-risk symptoms detected in the transcript."
        rec = "Advise monitoring and routine follow-up."

    return {
        "risk_level": risk,
        "symptoms": found_critical + found_high if (found_critical or found_high) else ["none detected"],
        "reason": reason,
        "recommendation": rec,
    }


def classify_risk(patient, transcript):
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return classify_with_claude(patient, transcript)
        except Exception as e:
            result = classify_with_rules(patient, transcript)
            result["reason"] += f" (AI classification failed, used rule-based fallback: {e})"
            return result
    return classify_with_rules(patient, transcript)


# ---------------------------------------------------------------------------
# Step 3: Emergency call processing (ties transcription + classification together)
# ---------------------------------------------------------------------------
def process_emergency_call(name, community, audio_path, manual_text):
    key = patient_key(name, community)
    patient = PATIENTS.get(key)

    if patient is None:
        return (
            f"No record found for {name}, {community}. Treating as unidentified caller.",
            "", "", dashboard_table()
        )

    transcript = manual_text.strip() if manual_text.strip() else transcribe_audio(audio_path)
    if not transcript:
        return "No audio or text provided.", "", "", dashboard_table()

    result = classify_risk(patient, transcript)

    alert = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": patient["name"],
        "community": patient["community"],
        "risk_level": result["risk_level"],
        "symptoms": ", ".join(result["symptoms"]),
        "reason": result["reason"],
        "recommendation": result["recommendation"],
        "transcript": transcript,
        "audio_path": audio_path,
    }
    ALERTS.insert(0, alert)

    summary = (
        f"RISK LEVEL: {result['risk_level']}\n\n"
        f"Symptoms: {', '.join(result['symptoms'])}\n\n"
        f"Reason: {result['reason']}\n\n"
        f"Recommendation: {result['recommendation']}"
    )
    return "Emergency report generated and sent to the dashboard.", transcript, summary, dashboard_table()


def dashboard_table():
    rows = [
        [a["time"], a["name"], a["community"], a["risk_level"], a["symptoms"], a["recommendation"]]
        for a in ALERTS
    ]
    return rows


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="SafeBirth Prototype") as demo:
    gr.Markdown("# SafeBirth Prototype\nRegistration -> Emergency Call -> AI Triage -> Healthcare Worker Dashboard")

    with gr.Tab("1. Patient Registration"):
        with gr.Row():
            name_in = gr.Textbox(label="Patient name")
            community_in = gr.Textbox(label="Community")
        with gr.Row():
            age_in = gr.Number(label="Age", value=28)
            gest_in = gr.Textbox(label="Gestational age", value="34 weeks")
        conditions_in = gr.Textbox(label="Medical history (e.g. hypertension, diabetes)")
        with gr.Row():
            lang_in = gr.Textbox(label="Preferred language", value="Twi")
            phone_in = gr.Textbox(label="Phone number (optional)")
        register_btn = gr.Button("Register patient", variant="primary")
        register_status = gr.Textbox(label="Status", interactive=False)
        register_list = gr.Dataframe(
            headers=["Name", "Community", "Age", "Gestational age", "Conditions", "Language"],
            label="Registered patients",
        )
        register_btn.click(
            register_patient,
            inputs=[name_in, community_in, age_in, gest_in, conditions_in, lang_in, phone_in],
            outputs=[register_status, register_list],
        )

    with gr.Tab("2. Emergency Call"):
        gr.Markdown("Identify the patient by registered name and community, then upload a voice recording "
                     "(or type a transcript directly to test without audio).")
        with gr.Row():
            call_name_in = gr.Textbox(label="Patient's registered name")
            call_community_in = gr.Textbox(label="Registered community")
        audio_in = gr.Audio(label="Upload emergency call recording", type="filepath")
        manual_text_in = gr.Textbox(label="Or type the transcript directly (for testing without audio)")
        call_btn = gr.Button("Process emergency call", variant="primary")
        call_status = gr.Textbox(label="Status", interactive=False)
        transcript_out = gr.Textbox(label="Transcript", interactive=False)
        classification_out = gr.Textbox(label="AI classification", interactive=False, lines=6)

    with gr.Tab("3. Healthcare Worker Dashboard"):
        gr.Markdown("Live emergency alerts, most recent first.")
        refresh_btn = gr.Button("Refresh dashboard")
        dashboard = gr.Dataframe(
            headers=["Time", "Name", "Community", "Risk level", "Symptoms", "Recommendation"],
            label="Emergency alerts",
        )
        refresh_btn.click(dashboard_table, outputs=dashboard)

    call_btn.click(
        process_emergency_call,
        inputs=[call_name_in, call_community_in, audio_in, manual_text_in],
        outputs=[call_status, transcript_out, classification_out, dashboard],
    )

if __name__ == "__main__":
    demo.launch()
