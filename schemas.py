from pydantic import BaseModel, Field
from typing import List

class ProjectSubmission(BaseModel):
    title: str
    problem_statement: str
    abstract: str
    objectives: str
    tech_stack: str
    methodology: str
    features: str
    expected_outcome: str

class CriteriaScore(BaseModel):
    innovation: int = Field(ge=0, le=100, description="Novelty and distinctiveness")
    feasibility: int = Field(ge=0, le=100, description="Technical achievability")
    impact: int = Field(ge=0, le=100, description="Real-world usefulness and social/business value")
    technical_depth: int = Field(ge=0, le=100, description="Complexity and architectural depth")
    scalability: int = Field(ge=0, le=100, description="Ease of expansion and scaling")
    presentation: int = Field(ge=0, le=100, description="Clarity and completeness")

class EvaluationResponse(BaseModel):
    overall_score: int
    scores: CriteriaScore
    strengths: List[str]
    weak_points: List[str]
    suggested_improvements: List[str]
    initial_judge_questions: List[str]

class VivaInitRequest(BaseModel):
    project_id: int
    judge_persona: str

class VivaJudgeStartResponse(BaseModel):
    judge_question: str
    defense_points: List[str] = Field(default_factory=list)

class VivaJudgeTurnResponse(BaseModel):
    verdict: str
    defense_points: int = Field(ge=0, le=10, description="Defense score out of 10")
    next_question: str

class VivaTurnRequest(BaseModel):
    session_id: int
    student_response: str