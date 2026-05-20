from pydantic import BaseModel, Field
from typing import List

class FindingSchema(BaseModel):
    summary: str = Field(description="The summary of the finding.")
    explanation: str = Field(description="An explanation of the finding.")

class CommunityReportSchema(BaseModel):
    title: str = Field(description="The title of the report summarizing the community.")
    summary: str = Field(description="An executive summary of the community's role and context.")
    findings: List[FindingSchema] = Field(description="A list of key findings and structured facts about the community.")
    rating: float = Field(description="A rating scale between 0-10 expressing how complete and impactful this report context is.")
    rating_explanation: str = Field(description="An explanation justifying the given rating based on the source data quality.")