from gtts import gTTS
import os

def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text, lang='en')  # 'en' for English
    tts.save(filename)
    print(f"Audio was saved as {filename}")

# Your provided text
text = """
Risk is the potential of losing something of value. 
Values can be gained or lost when taking risk resulting from a given action or inaction, foreseen or unforeseen. 
"""

# Function call to generate the MP3
text_to_speech(text)
