# python3 -m tests.structured_output

from openrouter.openrouter import *
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Participant(BaseModel):
    name: str = Field(..., description="Name of the participant")
    email: str = Field(..., description="Email address of the participant")
    role: Optional[str] = Field(
        default="attendee", 
        description="Role of the participant in the meeting (e.g. host, speaker, attendee)"
    )

class Location(BaseModel):
    name: str = Field(..., description="Name of the meeting location (physical or virtual)")
    address: str = Field(..., description="Full address of the location")
    online: bool = Field(False, description="Indicates if the meeting is online")
    link: Optional[str] = Field(
        default=None,
        description="URL link for the online meeting (used only if online is True)"
    )

class MeetingSettings(BaseModel):
    recorded: bool = Field(..., description="Whether the meeting will be recorded")
    allow_guest: bool = Field(..., description="Whether guests (not in the participant list) are allowed")
    max_duration_minutes: int = Field(
        ..., 
        gt=0,
        description="Maximum allowed duration of the meeting in minutes"
    )

class MeetingEvent(BaseModel):
    title: str = Field(..., description="Title or subject of the meeting")
    participants: List[Participant] = Field(..., description="List of all participants in the meeting")
    location: Location = Field(..., description="Location details of the meeting")
    settings: MeetingSettings = Field(..., description="Configuration and rules for the meeting")
    tags: Optional[List[str]] = Field(default_factory=list, description="Optional tags or labels for the meeting")


ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="Introduce yourself, please.")
response:MeetingEvent = ai.structured_output(model=gpt_5_mini, query=query, json_schema=MeetingEvent)
print(response)

