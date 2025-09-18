# python3 -m tests.structured_output

from openrouter.openrouter import *
from pydantic import BaseModel, Field
from typing import List, Optional


class StatItem(BaseModel):
    number: str = Field(
        description="Large numeric value to display prominently (e.g., '85%', '150M', '2.5K')"
    )
    label: str = Field(
        description="Short category or metric name for the statistic (e.g., 'Success Rate', 'Users', 'Revenue')"
    )
    description: str = Field(
        description="Brief explanatory text providing context for the statistic (1-2 sentences)"
    )


class Template1Data(BaseModel):
    title: str = Field(
        description="HTML page title that appears in browser tab"
    )
    main_title: str = Field(
        description="Large prominent heading text displayed at the top of the slide"
    )
    subtitle: str = Field(
        description="Secondary heading text that appears below the main title"
    )
    description: str = Field(
        description="Main body text paragraph that provides detailed explanation or context (2-3 sentences recommended)"
    )
    additional_text: str = Field(
        description="Secondary body text paragraph for additional information or supporting details (1-2 sentences)"
    )
    stats: List[StatItem] = Field(
        description="List of key statistics or metrics to display on the right side of the slide (2-4 items recommended)"
    )


ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="Tell me about yourself as if you were a person with a name, age, occupation, and hobbies.")
response: Template1Data = ai.structured_output(model=gpt_5_mini, query=query, json_schema=Template1Data)
print(response)

query = Message(text="Use different persona and Tell me again.")
response: Template1Data = ai.structured_output(model=gpt_5_mini, query=query, json_schema=Template1Data)
print(response)

