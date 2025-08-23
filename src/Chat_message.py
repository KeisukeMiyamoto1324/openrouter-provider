from .LLMs import LLMModel

from enum import Enum
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass

from openai.types.chat import ChatCompletion


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
    result: any = ""


class Chat_message:
    def __init__(self, 
                 text: str, 
                 images: list[Image.Image]=None, 
                 role: Role=Role.user, 
                 answerdBy: LLMModel=None, 
                 raw_response: ChatCompletion=None
                ) -> None:
        self.role = role
        self.text = text
        self.images = self._process_image(images=images)
        self.answeredBy: LLMModel = answerdBy
        
        self.tool_calls: list[ToolCall] = []
        self.raw_resoonse: ChatCompletion = raw_response


    def __str__(self) -> str:
        # ANSI color codes for blue, green, and reset (to default)
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
        
        # Append text and reset color formatting at the end
        message += self.text + RESET + "\n"
        
        return message
        
    def _process_image(self, images: list):
        """
        Process a list of images by resizing them to maintain aspect ratio and then converting them to base64 format.
        
        Args:
            images (list): A list of image objects to be processed.

        Returns:
            list: A list of base64-encoded image strings if input is not None/empty, otherwise `None`.
        
        Note:
            - Images should be provided as a "list" even if there is only a single image to process.
        """
        if images == None:
            return None

        base64_images = []
        for image in images:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            
            image = self._resize_image_aspect_ratio(image=image)
            image = self._convert_to_base64(image=image)
            base64_images.append(image)

        return base64_images
        
    def _convert_to_base64(self, image: Image) -> str:
        """
        Convert an image to a base64-encoded string.

        Args:
            image (Image): The image object to be converted to base64 format.

        Returns:
            str: The base64-encoded string representation of the image.

        Note:
            - The image format will default to 'JPEG' if the format is not specified.
        """
        buffered = BytesIO()
        format = image.format if image.format else 'JPEG'
        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64
    
    def _resize_image_aspect_ratio(self, image: Image, target_length=1024):
        """
        Resize an image to a target length while maintaining its aspect ratio.

        Args:
            image (Image): The image object to be resized.
            target_length (int, optional): The target length for the larger dimension (default is 1024).

        Returns:
            Image: The resized image object with maintained aspect ratio.

        Note:
            - The smaller dimension is scaled proportionally based on the larger dimension to maintain aspect ratio.
            - If the image's aspect ratio is non-square, the target_length is applied to the larger dimension.
        """
        
        width, height = image.size
        
        if width > height:
            new_width = target_length
            new_height = int((target_length / width) * height)
        else:
            new_height = target_length
            new_width = int((target_length / height) * width)

        resized_image = image.resize((new_width, new_height))
        
        return resized_image
    
        