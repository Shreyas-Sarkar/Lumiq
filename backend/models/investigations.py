"""Version 1 contracts for trusted analytical operations. No executable expressions."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Filter(StrictModel):
    column: str = Field(max_length=128)
    operator: Literal['eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in'] = 'eq'
    value: str | float | bool | list[str | float]

class AnalysisStep(StrictModel):
    operation: Literal['aggregate', 'group_by', 'compare_periods', 'correlation']
    metric: str = Field(max_length=128)
    aggregation: Literal['sum', 'mean', 'count', 'min', 'max', 'median'] = 'sum'
    group_by: str | None = None
    date_column: str | None = None
    second_metric: str | None = None
    periods: list[list[str]] = Field(default_factory=list, max_length=2)
    filters: list[Filter] = Field(default_factory=list, max_length=8)
    limit: int = Field(default=20, ge=1, le=50)
    sort_descending: bool = True

class AnalysisPlan(StrictModel):
    version: Literal[1] = 1
    title: str = Field(default='Dataset investigation', max_length=160)
    explanation: str = Field(default='', max_length=1000)
    clarification: str | None = Field(default=None, max_length=500)
    concept_answer: str | None = Field(default=None, max_length=2000)
    assumptions: list[str] = Field(default_factory=list, max_length=8)
    steps: list[AnalysisStep] = Field(default_factory=list, max_length=4)

    @model_validator(mode='after')
    def one_route(self):
        if sum(bool(x) for x in (self.steps, self.clarification, self.concept_answer)) != 1:
            raise ValueError('Choose calculations, a clarification, or a conceptual answer')
        return self

class Finding(StrictModel):
    evidence_id: str
    observation: str = Field(max_length=600)
    hypothesis: str | None = Field(default=None, max_length=400)

class Findings(StrictModel):
    findings: list[Finding] = Field(max_length=8)
    follow_ups: list[str] = Field(default_factory=list, max_length=3)

class InvestigationRequest(StrictModel):
    chat_id: str
    question: str = Field(min_length=1, max_length=1000)
    request_id: str = Field(min_length=8, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    parent_id: str | None = None
    plan: AnalysisPlan | None = None

class ResumeRequest(StrictModel):
    clarification: str | None = Field(default=None, max_length=1000)
    plan: AnalysisPlan | None = None
