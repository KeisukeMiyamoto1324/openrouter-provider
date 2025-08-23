# python3 -m tests.image

from src.openrouter import Message, OpenRouterClient
from src.llms import gpt_4o_mini
from PIL import Image


dog = Image.open("tests/images/dog.jpg")
cat = Image.open("tests/images/cat.jpg")

ai = OpenRouterClient(system_prompt="Please answer in English.")
query = Message(text="What can you see in the images?", images=[dog, cat])
response = ai.invoke(model=gpt_4o_mini, query=query)
ai.print_memory()

