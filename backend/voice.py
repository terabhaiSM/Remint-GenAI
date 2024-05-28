from openai import OpenAI
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()


def create_voice(input, voice_file_name):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input,
    )

    response.stream_to_file(f"sounds/{voice_file_name}.mp3")
    return f"sounds/{voice_file_name}.mp3"
    