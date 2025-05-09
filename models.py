from pydantic import BaseModel, Field
from typing import List, Union, TypedDict, Annotated, Tuple
import operator

class Plan(BaseModel):
    """Plan for the agent"""
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")


class PlanExecute(BaseModel):
    input: str | None = None
    plan: List[str] = []
    past_steps: Annotated[List[Tuple], operator.add] = []
    response: str | None = None

class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
