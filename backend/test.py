from flask import Flask
from dotenv import load_dotenv
from llama2 import extract_text_from_pdf
from app import generate_video  # Assuming generate_video function is defined in generate_video.py

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def test_generate_from_pdf(pdf_path):
    # Simulate handling a PDF upload and generating prompts
    try:
        # Extract content from PDF
        content = extract_text_from_pdf(pdf_path)
        print(content)
        # Assuming you generate story prompts and details based on the extracted content
        # Modify as per your actual prompt generation logic
        topic = "Sample Topic"
        flavor = "Educational"
        description = content  # Use the entire content for generating prompts

        # Generate video based on extracted prompts
        video_path = generate_video(topic, pdf_path, flavor, description=description)

        return video_path
    except Exception as e:
        return str(e)

# Example usage:
if __name__ == '__main__':
    pdf_path = 'path_to_your_pdf_file.pdf'  # Replace with the actual path to your PDF file
    result = test_generate_from_pdf(pdf_path)
    print(f"Generated video at: {result}")
