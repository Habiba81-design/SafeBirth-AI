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
import tempfile
import re

# ---------------------------------------------------------------------------
# In-memory "database". Swap this for Firebase/Supabase/Airtable later,
# the rest of the pipeline does not need to change.
# ---------------------------------------------------------------------------
PATIENTS = {}   # key: "name|community" -> patient record dict
ALERTS = []     # list of alert dicts, newest first


def patient_key(name, community):
    return f"{name.strip().lower()}|{community.strip().lower()}"


def register_patient(name, community, age, gestational_age, conditions, language, phone, address):
    if not name or not community:
        return "Name and community are required.", registration_table()

    key = patient_key(name, community)

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
        "address": address,
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

_whisper_processor = None
_whisper_model = None

def get_whisper():
    global _whisper_processor, _whisper_model
    if _whisper_model is None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        model_id = "openai/whisper-tiny"
        _whisper_processor = WhisperProcessor.from_pretrained(model_id)
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    return _whisper_processor, _whisper_model


def transcribe_audio(audio_path):
    if audio_path is None:
        return ""
    import soundfile as sf
    import numpy as np
    import torch
    from scipy.signal import resample as scipy_resample

    processor, model = get_whisper()
    speech, sr = sf.read(audio_path)

    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    if sr != 16000:
        num_samples = int(len(speech) * 16000 / sr)
        speech = scipy_resample(speech, num_samples)

    inputs = processor(speech.astype(np.float32), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0].strip()

_ga_asr_processor = None
_ga_asr_model = None

def get_ga_asr():
    global _ga_asr_processor, _ga_asr_model
    if _ga_asr_model is None:
        from transformers import AutoProcessor, AutoModelForCTC
        model_id = "KhayaAI/w2v-bert-gaa"
        _ga_asr_processor = AutoProcessor.from_pretrained(model_id)
        _ga_asr_model = AutoModelForCTC.from_pretrained(model_id)
    return _ga_asr_processor, _ga_asr_model


def transcribe_audio_ga(audio_path):
    if audio_path is None:
        return ""
    import soundfile as sf
    import numpy as np
    import torch
    from scipy.signal import resample as scipy_resample

    processor, model = get_ga_asr()
    speech, sr = sf.read(audio_path)

    if speech.ndim > 1:
        speech = speech.mean(axis=1)

    if sr != 16000:
        num_samples = int(len(speech) * 16000 / sr)
        speech = scipy_resample(speech, num_samples)

    inputs = processor(speech.astype(np.float32), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


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

    # Any of these pre-existing conditions raises the urgency of an otherwise
    # moderate symptom, since they make complications more likely or more
    # dangerous. Extend this list as your team reviews real clinical guidance.
    high_risk_conditions = ["hypertension", "diabetes", "preeclampsia",
                             "previous complications", "anemia", "eclampsia"]

    found_critical = [term for term in critical_terms if term in t]
    found_high = [term for term in high_terms if term in t]
    has_high_risk_condition = any(c in conditions for c in high_risk_conditions)

    if found_critical:
        risk = "CRITICAL"
        reason = f"Reported symptoms ({', '.join(found_critical)}) indicate an obvious life-threatening emergency."
        rec = "Dispatch emergency response immediately, do not wait for further assessment."
    elif found_high and has_high_risk_condition:
        matched_conditions = [c for c in high_risk_conditions if c in conditions]
        risk = "HIGH"
        reason = (f"Reported symptoms ({', '.join(found_high)}) combined with a history of "
                  f"{', '.join(matched_conditions)} indicate elevated risk.")
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
def process_emergency_call(name, community, audio_path, manual_text, current_location):
    key = patient_key(name, community)
    patient = PATIENTS.get(key)

    if audio_path is None and not manual_text.strip():
        return (
            "No recording detected yet. Wait a moment after stopping the recording, "
            "confirm the audio player shows a waveform, then try again.",
            "", "", dashboard_table()
        )

    # if manual_text.strip():
    #     transcript = manual_text.strip()
    # else:
    #     patient_language = (patient.get("language") if patient else "").strip().lower()
    #     if patient_language in ("ga", "gaa"):
    #         transcript = transcribe_audio_ga(audio_path)
    #     else:
    #         transcript = transcribe_audio(audio_path)
    if manual_text.strip():
        transcript = manual_text.strip()
    else:
        # Ga-language transcription (transcribe_audio_ga) is built and
        # verified working, but temporarily disabled here to keep memory usage
        # within Render's free tier limit for this submission. Re-enable by
        # restoring the language check below once running on higher-memory
        # hardware.
        transcript = transcribe_audio(audio_path)

    if not transcript:
        return "No audio or text provided.", "", "", dashboard_table()

    if patient is None:
        fallback_patient = {"name": name or "Unregistered caller", "community": community or "Unknown", "conditions": ""}
        result = classify_risk(fallback_patient, transcript)
        result["reason"] = f"No registration record found, classified from reported symptoms alone. {result['reason']}"

        alert = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": f"{name or 'Unknown'} (UNREGISTERED)",
            "community": community or "Unknown",
            "risk_level": result["risk_level"],
            "symptoms": ", ".join(result["symptoms"]),
            "reason": result["reason"],
            "recommendation": result["recommendation"],
            "transcript": transcript,
            "audio_path": audio_path,
            "current_location": current_location or "Not provided",
        }
        ALERTS.insert(0, alert)

        summary = (
            f"RISK LEVEL: {result['risk_level']} (no patient record matched)\n\n"
            f"Symptoms: {', '.join(result['symptoms'])}\n\n"
            f"Reason: {result['reason']}\n\n"
            f"Recommendation: {result['recommendation']}"
        )
        return "No matching registration found. Classified from symptoms alone and sent to dashboard.", transcript, summary, dashboard_table()

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
        "current_location": current_location or "Not provided",
    }
    ALERTS.insert(0, alert)

    summary = (
        f"RISK LEVEL: {result['risk_level']}\n\n"
        f"Symptoms: {', '.join(result['symptoms'])}\n\n"
        f"Reason: {result['reason']}\n\n"
        f"Recommendation: {result['recommendation']}"
    )
    return "Emergency report generated and sent to the dashboard.", transcript, summary, dashboard_table()

RISK_PRIORITY = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}


def dashboard_table():
    # ALERTS is already newest-first (we insert at index 0), so a stable
    # sort by risk level alone naturally preserves newest-first ordering
    # within each risk tier, no second sort key needed.
    sorted_alerts = sorted(ALERTS, key=lambda a: RISK_PRIORITY.get(a["risk_level"], 99))

    flag_map = {"CRITICAL": "\U0001F6A8 ", "HIGH": "\u26A0\uFE0F "}

    rows = []
    for a in sorted_alerts:
        flag = flag_map.get(a["risk_level"], "")
        rows.append([
            flag + a["time"],
            a["name"],
            a["community"],
            a["risk_level"],
            a["symptoms"],
            a["recommendation"],
        ])
    return rows

def alert_choices():
    return [f"{a['time']} | {a['name']}, {a['community']} | {a['risk_level']}" for a in ALERTS]


def generate_report(selected):
    if not selected:
        return "Select an alert above first."

    for a in ALERTS:
        label = f"{a['time']} | {a['name']}, {a['community']} | {a['risk_level']}"
        if label == selected:
            patient = PATIENTS.get(patient_key(a["name"].replace(" (UNREGISTERED)", ""), a["community"]))
            history_lines = ""
            if patient:
                history_lines = (
                    f"- Age: {patient.get('age', 'Not recorded')}\n"
                    f"- Gestational age: {patient.get('gestational_age', 'Not recorded')}\n"
                    f"- Known conditions: {patient.get('conditions') or 'None recorded'}\n"
                    f"- Address: {patient.get('address') or 'Not recorded'}\n"
                )
            else:
                history_lines = "- No registration record matched this caller.\n"

            report = (
                f"# Emergency Report\n\n"
                f"**Patient:** {a['name']}\n"
                f"**Community:** {a['community']}\n"
                f"**Call time:** {a['time']}\n"
                f"**Current location:** {a.get('current_location', 'Not provided')}\n\n"
                f"## Patient History\n{history_lines}\n"
                f"## Risk Assessment\n"
                f"**Risk level:** {a['risk_level']}\n\n"
                f"**Symptoms reported:** {a['symptoms']}\n\n"
                f"**Reasoning:** {a['reason']}\n\n"
                f"**Recommendation:** {a['recommendation']}\n\n"
                f"## Call Transcript\n\"{a['transcript']}\"\n\n"
                f"---\n"
                f"*This report was generated by an AI-assisted triage system. "
                f"It supports, but does not replace, clinical judgement.*"
            )
            return report

    return "Could not find that alert, it may have been cleared."


def generate_report_file(selected):
    report_text = generate_report(selected)
    if report_text.startswith("Select an alert") or report_text.startswith("Could not find"):
        return None

    html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>SafeBirth Emergency Report</title>
<style>
  body { font-family: Georgia, serif; max-width: 700px; margin: 40px auto; color: #222; line-height: 1.6; padding: 0 20px; }
  h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { font-size: 16px; margin-top: 24px; color: #444; }
  .disclaimer { font-style: italic; color: #666; font-size: 13px; margin-top: 30px; border-top: 1px solid #ccc; padding-top: 10px; }
  @media print { body { margin: 0; } }
</style>
</head>
<body>
"""
    for line in report_text.split("\n"):
        line = line.strip()
        line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)
        if line.startswith("# "):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith("## "):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith("---"):
            continue
        elif line.startswith("*") and line.endswith("*") and len(line) > 2:
            html_content += f'<p class="disclaimer">{line.strip("*")}</p>\n'
        elif line:
            html_content += f"<p>{line}</p>\n"

    html_content += "</body></html>"

    file_path = os.path.join(tempfile.gettempdir(), "safebirth_report.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return file_path
def clear_patients():
    PATIENTS.clear()
    return "All patient records cleared.", registration_table()


def clear_alerts():
    ALERTS.clear()
    return dashboard_table()

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
            age_in = gr.Number(label="Age", placeholder="Input your age")
            gest_in = gr.Textbox(label="Gestational age", placeholder="How many weeks?")
        conditions_in = gr.Textbox(label="Medical history (e.g. hypertension, diabetes)")
        with gr.Row():
            lang_in = gr.Textbox(label="Preferred language", value="English")
            phone_in = gr.Textbox(label="Phone number (optional)")
            address_in = gr.Textbox(label="Address / landmark (e.g. near the market, blue house)")
        register_btn = gr.Button("Register patient", variant="primary")
        register_status = gr.Textbox(label="Status", interactive=False)
        register_list = gr.Dataframe(
            headers=["Name", "Community", "Age", "Gestational age", "Conditions", "Language"],
            label="Registered patients",
        )
        clear_patients_btn = gr.Button("Clear all patient records")
        clear_patients_btn.click(clear_patients, outputs=[register_status, register_list])
        register_btn.click(
            register_patient,
            inputs=[name_in, community_in, age_in, gest_in, conditions_in, lang_in, phone_in, address_in],
            outputs=[register_status, register_list],
        )

    with gr.Tab("2. Emergency Call"):
        gr.Markdown("Identify the patient by registered name and community, then upload a voice recording "
                     "(or type a transcript directly to test without audio).")
        with gr.Row():
            call_name_in = gr.Textbox(label="Patient's registered name")
            call_community_in = gr.Textbox(label="Registered community")
        current_location_in = gr.Textbox(label="Current location")
        audio_in = gr.Audio(label="Upload emergency call recording", type="filepath")
        manual_text_in = gr.Textbox(label="Or type the transcript directly (for testing without audio)")
        call_btn = gr.Button("Process emergency call", variant="primary")
        call_status = gr.Textbox(label="Status", interactive=False)
        transcript_out = gr.Textbox(label="Transcript", interactive=False)
        classification_out = gr.Textbox(label="AI classification", interactive=False, lines=6)

    with gr.Tab("3. Healthcare Worker Dashboard"):
        gr.Markdown("Live emergency alerts, most recent first.")
        with gr.Row():
            with gr.Column(scale=3):
                refresh_btn = gr.Button("Refresh dashboard")
                dashboard = gr.Dataframe(
                    headers=["Time", "Name", "Community", "Risk level", "Symptoms", "Recommendation"],
                    label="Emergency alerts",
                )
                clear_alerts_btn = gr.Button("Clear all alerts")

            with gr.Column(scale=2):
                gr.Markdown("### Generate a report for a doctor")
                report_selector = gr.Dropdown(choices=[], label="Select an alert")
                generate_report_btn = gr.Button("Generate report", variant="primary")
                download_report_btn = gr.Button("Download printable report")
                report_output = gr.Markdown()
                report_file = gr.File(label="Printable report", visible=False)
                report_output = gr.Markdown()

        refresh_btn.click(dashboard_table, outputs=dashboard).then(lambda: gr.update(choices=alert_choices()), outputs=report_selector)
        clear_alerts_btn.click(clear_alerts, outputs=dashboard).then(lambda: gr.update(choices=alert_choices()), outputs=report_selector)
        generate_report_btn.click(generate_report, inputs=report_selector, outputs=report_output)
        download_report_btn.click(generate_report_file, inputs=report_selector, outputs=report_file).then(
            lambda f: gr.update(visible=f is not None), inputs=report_file, outputs=report_file
        )

    call_btn.click(
        process_emergency_call,
        inputs=[call_name_in, call_community_in, audio_in, manual_text_in, current_location_in],
        outputs=[call_status, transcript_out, classification_out, dashboard],
    ).then(lambda: gr.update(choices=alert_choices()), outputs=report_selector)

    demo.load(fn=lambda: gr.update(choices=alert_choices()), outputs=report_selector)
import threading
import time
import requests


def keep_alive():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return  # not running on Render, skip entirely (e.g. running locally)
    while True:
        time.sleep(600)  # every 10 minutes, safely under Render's 15-minute idle limit
        try:
            requests.get(url, timeout=10)
        except Exception:
            pass  # a single failed ping shouldn't crash this background loop

if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

