"""
PaperX content schema. Every paper is data, not a hand-built page.
Paper #2 should take hours, not days, because it only fills this structure.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Beat = Literal["problem", "idea", "mechanism", "proof", "impact"]


class Source(BaseModel):
    """Where in the original paper a claim comes from."""
    page: int
    para_idx: Optional[int] = None
    quote: Optional[str] = Field(None, description="Short verbatim span from the paper")
    verified: bool = Field(False, description="True only after you checked it against the PDF")


class CodeLayer(BaseModel):
    pytorch: str                       # the snippet; tests exec this exact string
    shapes: dict[str, str] = {}        # e.g. {"q, k": "[B, T, d_k]"}


class ThenVsNow(BaseModel):
    paper: str = Field(..., description="What this paper did")
    now: str = Field(..., description="What modern practice does differently")


class Analogy(BaseModel):
    text: str = Field(..., description="The analogy, 1-2 sentences")
    breaks: str = Field(..., description="Where the analogy stops being accurate")


class Section(BaseModel):
    id: str
    beat: Beat
    title: str
    # Progressive depth: shallow by default, deeper on demand
    intuition: str = Field(..., description="Plain words, ~80 words max")
    mechanism: Optional[str] = Field(None, description="Plain-text explanation of how it works")
    equation: Optional[str] = Field(None, description="LaTeX, rendered with KaTeX")
    code: Optional[CodeLayer] = None
    then_vs_now: Optional[ThenVsNow] = None
    analogy: Optional[Analogy] = None
    next_teaser: Optional[str] = Field(None, description="Cliffhanger shown before Next")
    think_first: Optional[str] = Field(None, description="Question to ponder before reveal")
    think_first_answer: Optional[str] = None
    visuals: list[str] = Field([], description="Keys of visual components shown in this section")
    sources: list[Source] = []


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    year: int
    arxiv_url: str
    hook: str = Field(..., description="The problem the authors were stuck on")
    read_minutes: int
    sections: list[Section]
    unlocked: list[str] = Field([], description="What came after (BERT, GPT...)")