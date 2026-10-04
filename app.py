import os
import json
import time
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, ListFlowable

from schemas import (
    ProjectSubmission,
    EvaluationResponse,
    VivaInitRequest,
    VivaJudgeStartResponse,
    VivaJudgeTurnResponse,
    VivaTurnRequest
)
import database as db

load_dotenv()
db.init_db()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment or .env file.")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.8-flash"


def generate_with_retry(prompt: str, *, temperature: float = 0.2, response_schema=None, system_instruction=None):
    last_error = None
    for attempt in range(3):
        try:
            config_kwargs = {"temperature": temperature}
            if response_schema is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = response_schema
            if system_instruction is not None:
                config_kwargs["system_instruction"] = system_instruction

            config = types.GenerateContentConfig(**config_kwargs)
            return client.models.generate_content(model=MODEL_NAME, contents=prompt, config=config)
        except Exception as e:
            last_error = e
            msg = str(e)
            if "503" in msg or "429" in msg or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            raise
    raise RuntimeError(f"Gemini request failed: {last_error}")


def fallback_evaluation(submission: ProjectSubmission):
    text = f"{submission.title} {submission.problem_statement} {submission.abstract} {submission.objectives} {submission.tech_stack} {submission.methodology} {submission.expected_outcome}".lower()
    score = 62
    if len(submission.abstract) > 120:
        score += 8
    if "ai" in text or "ml" in text or "computer vision" in text:
        score += 6
    if "dataset" in text or "testing" in text:
        score += 4
    if "deployment" in text or "system" in text:
        score += 5
    score = max(0, min(100, score))

    return {
        "overall_score": score,
        "scores": {
            "innovation": max(0, min(100, score - 8)),
            "feasibility": max(0, min(100, score - 5)),
            "impact": max(0, min(100, score - 3)),
            "technical_depth": max(0, min(100, score - 7)),
            "scalability": max(0, min(100, score - 10)),
            "presentation": max(0, min(100, score - 4)),
        },
        "strengths": [
            "The project addresses a real problem with a practical solution.",
            "The technical stack and workflow are clearly defined.",
            "The project shows a good foundation for further engineering work.",
        ],
        "weak_points": [
            "The system design would benefit from clearer validation metrics and benchmarks.",
            "More discussion is needed on edge cases, failure handling, and scalability.",
            "The novelty claim should be supported with stronger comparative analysis.",
        ],
        "suggested_improvements": [
            "Add a clearer dataset and evaluation methodology.",
            "Document technical trade-offs and failure scenarios.",
            "Include a baseline comparison and deployment plan.",
        ],
        "initial_judge_questions": [
            "How did you validate the core technical assumptions?",
            "What happens when the system receives noisy or unexpected input?",
            "What is your fallback plan if the primary solution fails under load?",
        ],
    }


def fallback_viva_start(project_title: str, judge_persona: str):
    base_question = f"How did you validate the core technical assumptions in {project_title} under realistic constraints?"
    persona_points = {
        "technical": [
            "Explain the design trade-offs in your architecture.",
            "Show how you handled edge cases and failure modes.",
            "Quantify the performance and validation evidence.",
        ],
        "innovation": [
            "Explain what makes the project genuinely novel.",
            "Compare your approach against existing solutions.",
            "Show where the project creates clear value beyond a demo.",
        ],
        "industry": [
            "Describe the cost, deployment, and scalability model.",
            "Explain the real-world adoption constraints.",
            "Justify the business and operational value of the system.",
        ],
    }
    judge_points = persona_points.get(judge_persona, [
        "Explain the technical reasoning behind the design.",
        "Show evidence that the system works under realistic conditions.",
        "Describe the most important trade-offs and risks.",
    ])
    return {"judge_question": base_question, "defense_points": judge_points[:3]}


def fallback_viva_turn(student_response: str, project_title: str):
    lower = student_response.lower()
    technical_strength = any(word in lower for word in ["architecture", "tested", "dataset", "performance", "latency", "scalability", "model", "validation", "benchmark", "trade-off", "tradeoff"])
    score = 6 if technical_strength else 4
    if len(student_response) > 180:
        score += 2
    score = min(10, max(0, score))
    verdict = "The answer is technically grounded and credible." if technical_strength else "The answer is still shallow and needs deeper technical justification."
    next_question = f"What evidence proves the core assumptions of {project_title} survive real-world constraints and failure cases?"
    return {"verdict": verdict, "defense_points": score, "next_question": next_question}


def build_project_report_pdf(project_data: dict, evaluation_data: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title = project_data.get("title", "Untitled Project")
    story.append(Paragraph("Project Report", styles["Title"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph(f"<b>Project Title:</b> {title}", styles["Heading2"]))
    story.append(Spacer(1, 10))

    field_map = [
        ("Problem Statement", project_data.get("problem_statement", "")),
        ("Abstract", project_data.get("abstract", "")),
        ("Objectives", project_data.get("objectives", "")),
        ("Tech Stack", project_data.get("tech_stack", "")),
        ("Methodology", project_data.get("methodology", "")),
        ("Features", project_data.get("features", "")),
        ("Expected Outcome", project_data.get("expected_outcome", "")),
    ]

    for label, value in field_map:
        if value:
            story.append(Paragraph(f"<b>{label}:</b> {value}", styles["BodyText"]))
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Overall Score:</b> {evaluation_data.get('overall_score', 'N/A')}/100", styles["Heading2"]))

    scores = evaluation_data.get("scores", {})
    if scores:
        score_lines = [
            f"Innovation: {scores.get('innovation', 'N/A')}",
            f"Feasibility: {scores.get('feasibility', 'N/A')}",
            f"Impact: {scores.get('impact', 'N/A')}",
            f"Technical Depth: {scores.get('technical_depth', 'N/A')}",
            f"Scalability: {scores.get('scalability', 'N/A')}",
            f"Presentation: {scores.get('presentation', 'N/A')}",
        ]
        story.append(Paragraph("<b>Criteria Scores:</b>", styles["Heading2"]))
        story.append(ListFlowable([Paragraph(line, styles["BodyText"]) for line in score_lines], bulletType='bullet', bulletText='•'))

    if evaluation_data.get("strengths"):
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Strengths:</b>", styles["Heading2"]))
        story.append(ListFlowable([Paragraph(item, styles["BodyText"]) for item in evaluation_data["strengths"]], bulletType='bullet', bulletText='•'))

    if evaluation_data.get("weak_points"):
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Weak Points:</b>", styles["Heading2"]))
        story.append(ListFlowable([Paragraph(item, styles["BodyText"]) for item in evaluation_data["weak_points"]], bulletType='bullet', bulletText='•'))

    if evaluation_data.get("suggested_improvements"):
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Suggested Improvements:</b>", styles["Heading2"]))
        story.append(ListFlowable([Paragraph(item, styles["BodyText"]) for item in evaluation_data["suggested_improvements"]], bulletType='bullet', bulletText='•'))

    doc.build(story)
    return buffer.getvalue()


app = FastAPI(title="ProjectJudge AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.get("/api/report/pdf")
def get_project_report(project_id: int):
    sub_data, eval_data = db.get_project(project_id)
    if not sub_data:
        raise HTTPException(status_code=404, detail="Project not found")
    pdf_bytes = build_project_report_pdf(sub_data, eval_data)
    filename = f"{sub_data.get('title', 'project-report').replace(' ', '_')}.pdf"
    return Response(pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.post("/api/report/pdf")
def create_project_report(payload: dict):
    project_data = payload.get("project") or {}
    evaluation_data = payload.get("evaluation") or {}
    if not project_data:
        raise HTTPException(status_code=400, detail="Project data is required")
    pdf_bytes = build_project_report_pdf(project_data, evaluation_data)
    filename = f"{project_data.get('title', 'project-report').replace(' ', '_')}.pdf"
    return Response(pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.post("/api/evaluate")
def evaluate_project(submission: ProjectSubmission):
    prompt = f"""
    Evaluate the following student engineering project submission rigorously:

    Title: {submission.title}
    Problem Statement: {submission.problem_statement}
    Abstract: {submission.abstract}
    Objectives: {submission.objectives}
    Tech Stack: {submission.tech_stack}
    Methodology: {submission.methodology}
    Key Features: {submission.features}
    Expected Outcome: {submission.expected_outcome}

    Provide an objective, critical, and constructively harsh evaluation using the requested schema.
    """

    try:
        response = generate_with_retry(
            prompt,
            temperature=0.2,
            response_schema=EvaluationResponse,
            system_instruction=(
                "You are an expert capstone project judging panel composed of university professors and tech leads. "
                "Score objectively out of 100. Deduct points for missing datasets, vague architecture, absence of baseline comparisons, "
                "or unverified claims."
            ),
        )
        evaluation_data = json.loads(response.text)
    except Exception as e:
        evaluation_data = fallback_evaluation(submission)

    project_id = db.save_project(submission.model_dump(), evaluation_data)
    return {"project_id": project_id, "evaluation": evaluation_data}

PERSONA_PROMPTS = {
    "technical": "You are a Principal Software/Hardware Architect. Challenge system bottlenecks, data pipelines, failure modes, latency, and edge cases.",
    "innovation": "You are an R&D Lead. Challenge the originality of this project compared to existing open-source libraries, tutorials, and standard papers.",
    "industry": "You are a Tech VC & Product Director. Drill into unit economics, hosting costs, realistic market adoption, and scaling limits.",
    "viva": "You are a strict University Viva Examiner. Probe the student's personal depth of understanding, methodology choices, and basic fundamentals."
}

@app.post("/api/viva/start")
def start_viva(req: VivaInitRequest):
    sub_data, eval_data = db.get_project(req.project_id)
    if not sub_data:
        raise HTTPException(status_code=404, detail="Project not found")

    persona_desc = PERSONA_PROMPTS.get(req.judge_persona, PERSONA_PROMPTS["viva"])
    
    prompt = f"""
    {persona_desc}
    
    Project Title: {sub_data['title']}
    Tech Stack: {sub_data['tech_stack']}
    Methodology: {sub_data['methodology']}
    Identified Weaknesses: {', '.join(eval_data.get('weak_points', []))}
    
    Start the defense by asking ONE sharp, direct opening question to the student testing their project. Do not greet with pleasantries, jump straight to the technical question.
    Also provide exactly 3 short defense points the judge is looking for as a quick rubric for the student.
    """

    try:
        response = generate_with_retry(
            prompt,
            temperature=0.4,
            response_schema=VivaJudgeStartResponse,
        )
        payload = json.loads(response.text)
        opening_question = payload["judge_question"].strip()
        judge_points = payload.get("defense_points", [])[:3]
    except Exception as e:
        payload = fallback_viva_start(sub_data["title"], req.judge_persona)
        opening_question = payload["judge_question"].strip()
        judge_points = payload.get("defense_points", [])[:3]

    history = [{"role": "judge", "message": opening_question}]
    session_id = db.save_viva_session(req.project_id, req.judge_persona, history)
    
    return {"session_id": session_id, "judge_question": opening_question, "judge_points": judge_points}

@app.post("/api/viva/turn")
def viva_turn(req: VivaTurnRequest):
    session = db.get_viva_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Viva session not found")

    sub_data, _ = db.get_project(session["project_id"])
    persona_desc = PERSONA_PROMPTS.get(session["persona"], PERSONA_PROMPTS["viva"])

    history = session["history"]
    history.append({"role": "student", "message": req.student_response})

    convo_transcript = "\n".join([f"{item['role'].upper()}: {item['message']}" for item in history])

    prompt = f"""
    {persona_desc}

    Project Context:
    Title: {sub_data['title']}
    Tech Stack: {sub_data['tech_stack']}

    Conversation Transcript:
    {convo_transcript}

    Evaluate the student's latest response:
    1. In 1 sentence, give a verdict on whether they answered accurately or dodged the technical core.
    2. Assign a defense score from 0 to 10 based on how strong the answer is.
    3. Then ask your next hard follow-up question or probe deeper into the trade-offs.
    Keep everything direct, concise, and professional.
    """

    try:
        response = generate_with_retry(
            prompt,
            temperature=0.4,
            response_schema=VivaJudgeTurnResponse,
        )
        payload = json.loads(response.text)
        verdict = payload["verdict"].strip()
        defense_points = int(payload.get("defense_points", 0))
        judge_reply = payload["next_question"].strip()
    except Exception as e:
        payload = fallback_viva_turn(req.student_response, sub_data['title'])
        verdict = payload["verdict"].strip()
        defense_points = int(payload.get("defense_points", 0))
        judge_reply = payload["next_question"].strip()

    history.append({"role": "judge", "message": judge_reply})
    db.update_viva_history(req.session_id, history)

    return {"judge_reply": judge_reply, "verdict": verdict, "defense_points": defense_points}