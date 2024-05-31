import os
import tempfile
import json
import PyPDF2
from llama_index.core import VectorStoreIndex, PromptHelper
from llama_index.llms.langchain import LangChainLLM
from app import generate_story_prompts, story_json
from llama import extract_content_from_pdf

# For testing purposes
if __name__ == "__main__":
    # Provide the path to your test PDF file here
    test_pdf_path = "data/psyco.pdf"
    title = "WHAT IS THIS THING CALLED PSYCHOLOGY?"
    description= extract_content_from_pdf("data", f"Provide a 200 words description for: {title} from the pdf")
    print(description)
    flavor = "Informational"
    system_prompt, user_prompt = generate_story_prompts(title, flavor, "Mental health", str(description))
    story = extract_content_from_pdf("data", f"Generate a story based on the {system_prompt} and {user_prompt} from the pdf")
    story =  json.loads(story_json(story))
    print(story)

