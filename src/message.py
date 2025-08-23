from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
import base64
from typing import Optional, Any

from PIL import Image
from openai.types.chat import ChatCompletion

from .llms import LLMModel


class Role(Enum):
    system = "system"
    user = "user"
    ai = "assistant"
    agent = "agent"
    tool = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    result: Any = ""


class Message:
    def __init__(
        self,
        text: str,
        images: Optional[list[Image.Image]] = None,
        role: Role = Role.user,
        answered_by: Optional[LLMModel] = None,
        raw_response: Optional[ChatCompletion] = None
    ) -> None:
        self.role = role
        self.text = text
        self.images = self._process_image(images)
        self.answered_by = answered_by
        self.tool_calls: list[ToolCall] = []
        self.raw_response = raw_response


    def __str__(self) -> str:
        BLUE = "\033[34m"
        GREEN = "\033[32m"
        RESET = "\033[0m"
        
        message = ""
        
        if self.role == Role.system:
            message = "---------------------- System ----------------------\n"
        elif self.role == Role.user:
            message = BLUE + "----------------------- User -----------------------\n" + RESET
        elif self.role == Role.ai:
            message = GREEN + "--------------------- Assistant --------------------\n" + RESET
        
        message += self.text + RESET + "\n"
        
        return message
        
    def _process_image(self, images: Optional[list[Image.Image]]) -> Optional[list[str]]:
        if images is None:
            return None

        base64_images = []
        for image in images:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            
            image = self._resize_image_aspect_ratio(image)
            image = self._convert_to_base64(image)
            base64_images.append(image)

        return base64_images
        
    def _convert_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        format_type = image.format if image.format else 'JPEG'
        image.save(buffered, format=format_type)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64
    
    def _resize_image_aspect_ratio(self, image: Image.Image, target_length: int = 1024) -> Image.Image:
        width, height = image.size
        
        if width > height:
            new_width = target_length
            new_height = int((target_length / width) * height)
        else:
            new_height = target_length
            new_width = int((target_length / height) * width)

        return image.resize((new_width, new_height))