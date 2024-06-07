from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import openai
from moviepy.editor import (
    ImageSequenceClip,
    CompositeVideoClip,
    AudioFileClip,
    VideoFileClip,
    concatenate_videoclips,
)
from gtts import gTTS
import os
import json
import requests
from llama import extract_content_from_pdf
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from generateClip import generate_clip
import cv2
import numpy as np
import whisper
# from llama_index.core.tools import QueryEngineTool
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

app = Flask(__name__, static_folder="../build", static_url_path="/")
CORS(app)

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def fetch_summary(pdf_path):
    print(pdf_path)
    summary = extract_content_from_pdf("uploads")
    print(summary)
    return summary


@app.route("/", methods=["POST"])
def index():
    data = request.form
    topic = data.get("topic")
    flavor = data.get("flavor")
    description = data.get("description")
    description_url = data.get("description_url")
    pdf = request.files.get("pdf")
    print(topic, flavor, description, description_url, pdf)

    # Assuming video_details is a dictionary with the expected keys
    try:
        video_details = generate_video(
            topic, pdf, flavor, description, description_url, 0
        )
        return jsonify({"success": True, "video_path": video_details})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/pdf", methods=["POST"])
def pdfFunction():
    data = request.form
    topic = data.get("topic")
    pdf = request.files.get("pdf")
    # flavor = data.get('flavor')
    print(topic, pdf)
    # Assuming video_details is a dictionary with the expected keys
    try:
        video_details = generate_video(topic, pdf, "Informational", None, None, 1)
        return jsonify({"success": True, "video_path": video_details})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def save_image_from_url(image_url, save_path):
    """
    Download an image from a URL and save it to a specified path.

    :param image_url: A string containing the URL of the image.
    :param save_path: A string representing the file path to save the image to.
    """
    # Send a GET request to the image URL
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Image saved to {save_path}")
    else:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")


# Will write code for this next, once everything else is done and working
def evaluate_story(story):
    pass


def therapeutic_voice_over(voice_over):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                You are a mental health professional specializing in therapeutic communication.
                When provided with a voice over script, you task is to rewrite it in a more therapeutic tone.
                Rewriting doesn't mean changing the script, but adding punctuations so that when the script converted to audio, it sounds more therapeutic.
                """,
            },
            {
                "role": "user",
                "content": f"Create a therapeutic version of this voice over script in single quotes-  '{voice_over}'",
            },
        ],
        temperature=0.8,
        max_tokens=100,  # Adjusted max_tokens to allow a more detailed response
        top_p=1,
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


def generate_caption_prompt(image_description, caption, call_to_action=None):
    """
    Generate a detailed prompt for an AI to add a caption to an image.

    :param image_description: A string describing the image content and style.
    :param caption: A string containing the main caption to add to the image.
    :param call_to_action: Optional; a string containing a call to action.
    :return: A string prompt for image modification.

    Example usage:
    image_desc = "a serene park with a person sitting quietly on a bench"
    main_caption = "Find Your Peace in Nature"
    cta_caption = "Join our Nature Walks"

    prompt = generate_caption_prompt(image_desc, main_caption, cta_caption)
    print(prompt)

    """

    # Start with describing the image and the desired placement of the caption
    prompt = f"Edit the image which is {image_description}, to include a caption. "

    # Add details about the main caption
    prompt += f"Overlay the caption '{caption}' in a prominent place on the image "
    prompt += (
        "using a semi-transparent text box. The font should be clear and legible, "
    )
    prompt += "complementing the overall aesthetic of the image."
    prompt += (
        "The text should be large enough to read easily but not overwhelm the image."
    )
    prompt += "Other than the given caption '{caption}', no other text should be present on the image."

    # If a call to action is provided, add that to the prompt
    if call_to_action:
        prompt += f" Include a call to action with the text '{call_to_action}' "
        prompt += "placed in a visually appealing button that is easy to see and aligns with the image's theme."

    return prompt


# Function to fetch description from a url


def fetch_description_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response.text
    except requests.RequestException as e:
        print(f"An error occurred while fetching the description: {e}")
        return None


def generate_story_prompts(
    topic, flavor, space=None, description=None, description_url=None
):
    system_prompt = ""
    base_prompt = f"""
                You are a marketing genius specializing in the {space} space, tasked with creating video ideas for short videos, when given a topic/description and a flavor by the user.
                Your job is to outline a complete video concept that can be visually represented through a series of 4-5 images, each displayed for 2-5 seconds.

                The video will focus on the video topic or the video description given by user.

                The video will maintain a flavor as the user has requested. 

                Video Direction:
        - Title and Description: Provide a catchy title and a brief description of the video's content.
        - Scene-by-Scene Breakdown: Describe each scene in detail, specifying the type of images to be used, and the impact you aim to achieve with each.
        - Captions and Voiceovers: Outline what text should be overlayed as captions on each image and what should be narrated in the voiceover. Ensure the language is simple and accessible for non-English speakers.
        - Image Prompts: For each scene, provide a detailed image generation prompt for DALL-E, including stylistic details to ensure consistency across all images. The image prompt should specify that no text should be present on the image other than the provided captions or call-to-action text explicitly.
"""
    # Not using user_prompt for now
    user_prompt = f"""Create a short video idea with the following details:
    [Topic]: [topic]
    [Description] : [description]
    Flavor: [flavor]
"""

    # Check if description URL is provided
    if description_url:
        fetched_description = fetch_description_from_url(description_url)
        if fetched_description:
            description = fetched_description

    # Insert description into the base prompt if provided through description or description_url
    if description:
        user_prompt = user_prompt.replace("[Description]", "Video Description")
        user_prompt = user_prompt.replace("[description]", description)
    else:
        user_prompt = user_prompt.replace("[Topic]", "Video Topic")
        user_prompt = user_prompt.replace("[topic]", topic)

    flavor_content = {
        "Meme": "The video should be crafted in a humorous style, mimicking the format of popular internet memes to engage the audience.",
        "Advanced Insights": "The video should provide deep insights suitable for advanced users, including technical details and professional analysis.",
        "Warning": "The video should have a strong impact, presenting the information with a shock value to grab the audience's attention immediately.",
        "Informational": "The video should be informative and educational, presenting the topic in a clear and concise manner suitable for a general audience.",
        "Psychoeducation": "The video should focus on psychoeducational content, providing valuable insights and practical tips for mental health management.",
        "Shock Value": "The video should be designed to evoke strong emotions and create a lasting impact on the viewer through shocking visuals and narratives.",
        "Virality": "The video should be crafted with the intention of going viral, incorporating elements that resonate with a wide audience and encourage sharing.",
        "Negative News": "The video should highlight negative aspects of the topic, presenting a critical view to raise awareness and spark discussions.",
        "Positive News": "The video should focus on positive news and uplifting stories related to the topic, inspiring hope and optimism in the viewers.",
    }

    # Insert flavor-specific content into the base prompt
    user_prompt = user_prompt.replace("[flavor]", "Educational")

    print(base_prompt)

    return system_prompt, user_prompt


def generate_story(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=1500,  # Adjusted max_tokens to allow a more detailed response
        top_p=1,
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


def generate_story_old(topic, flavor):
    print(topic, flavor)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                You are marketing genius in the mental health space that can create video ideas for short videos provided topic and flavor. 
                You make sure that the video can be created with 4-5 images where every image is played for 2-5 seconds. In your response you can give an entire direction for the video. The direction can consist of kind of images being played scene by scene to make an impact and what information can be overlayed as caption on the image and what information can be spoken in voice over. You can also give a title and description of the video.
                
                The images that you suggest must contain a detailed image generation prompt for DALL-E. 

                The image prompts for a scene should contain all the information within themselves.

                Do not refer to any other scenes like, the boy from scene-1 or the image in scene-1. If you want to refer to other scenes, mention it in the prompt in a way that the scene is illustrated completely.
                
                Caption and voiceovers will be used to show the text. 

                Include styling information in all image generation prompts.

                The prompts generated for images should be extremely detailed to generate good images that fit the story line suggested by you.

                For the sake of generating great story, generate at least 5 stories and see which one of them is good at:
                    A) Conveying the information
                    B) Images are easy to create when prompt is giving to dall-e

                Give the best story and the complete details.

                Only reply when given a topic and a flavor.

                Keep the narration and caption script very easy to understand for non-English speaking audiences too. Do not use heavy English words and keep it simple.

                In the image generation prompt you will include the styling information for all images and make sure that all of them follow similar styling.

                The image prompts must contain the image styling information and the prompt for the image. The prompt should be detailed enough to generate a good image.

                Please double check the prompt before submitting. It should contain all the information needed to generate the image and the style of the image. All the styles should be consistent across all images.
                """,
            },
            {
                "role": "user",
                "content": f"Create a short video idea for the topic {topic} and flavor is {flavor}.",
            },
        ],
        temperature=0.8,
        max_tokens=1500,  # Adjusted max_tokens to allow a more detailed response
        top_p=1,
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


def story_json(story):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                Given a story, you need to create a json object for the story.
                The json object should contain the following
                    {   "story_title" : "Title of the story",
                        "story_description" : "Description of the story",
                        "complete_voice_over" : "Add all the voice overs of the scenes here with fullstop at the end of each sentence",
                        "scenes" : [
                            {"image_prompt" : "Prompt for the image",
                            "caption" : "Caption to overlay on the image",
                            "voice_over" : "Voice over text for the scene"
                            }
                        ]
                        }
                The json should be valid and it should follow the above structure always.
                Think before answering and validate the answer before submitting.
                """,
            },
            {"role": "user", "content": f"Create a json for the story {story}."},
        ],
        temperature=0.8,
        max_tokens=1500,  # Adjusted max_tokens to allow a more detailed response
        top_p=1,
    )

    return response.choices[0].message.content


def generate_story_json(system_prompt, user_prompt):
    story = generate_story(system_prompt, user_prompt)
    story = json.loads(story_json(story))
    print(story)
    return story


def text_to_speech(text, story_title):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )

    response.stream_to_file(f"sounds/{story_title}_audio.mp3")
    return f"sounds/{story_title}_audio.mp3"


def create_main_image(image_prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    print(image_url)
    return image_url


def generate_video(
    topic, pdf, flavor, description=None, description_url=None, num=None
):
    if num == 0:
        system_prompt, user_prompt = generate_story_prompts(
            topic, flavor, "Mental health", str(description or summary), description_url
        )
        story = generate_story_json(system_prompt, user_prompt)
    else:
        test_pdf_path = "data/psyco.pdf"
        # title = "WHAT IS THIS THING CALLED PSYCHOLOGY?"
        # description= extract_content_from_pdf("data", f"Provide a 200 words description for: {title} from the pdf")
        # print(description)
        # flavor = "Informational"
        # system_prompt, user_prompt = generate_story_prompts(title, flavor, "Mental health", str(description))
        # story = extract_content_from_pdf("data", f"Generate a story based on the {system_prompt} and {user_prompt} from the pdf")
        # story =  json.loads(story_json(story))
        # print(story)
    # raise ValueError("Story JSON is not generated yet.")

    #     story = {
    #     "story_title": topic,
    #     "story_description": "This video explores how Cognitive Behavioral Therapy (CBT) can be an e ective tool for managing symptoms of Attention Deficit Hyperactivity Disorder (ADHD). Through simple explanations and illustrative examples, viewers will learn how CBT techniques can be tailored to help individuals with ADHD enhance their focus, organization, and coping strategies.",
    #     "complete_voice_over": "In a world where focus seems elusive and attention wanders, managing ADHD can feel like an uphill battle. But what if there's a way to navigate through it effectively? Enter Cognitive Behavioral Therapy, or CBT. This therapy doesn't just aim to understand your challenges; it equips you with practical tools to reshape your responses and reclaim your focus. Let's explore how CBT can make a difference.",
    #     "scenes": [
    #         {
    #             "image_prompt": "Illustration of a diverse classroom setting with a child struggling to focus while other kids are engaged, in a colorful, cartoon style.",
    #             "caption": "Struggling with focus? You're not alone.",
    #             "voice_over": "ADHD affects many, making focus and attention challenging. But, what if there's a way to manage it effectively?"
    #         },
    #         {
    #             "image_prompt": "Warm and inviting therapy room with a therapist and a young patient discussing over CBT tools like journals and emotion cards, in a realistic style.",
    #             "caption": "Discover Cognitive Behavioral Therapy.",
    #             "voice_over": "Cognitive Behavioral Therapy, or CBT, is a type of therapy that helps manage your responses by changing thought and behavior patterns."
    #         },
    #         {
    #             "image_prompt": "Animated diagram showing a cycle of thoughts, emotions, and behaviors with arrows indicating changes through CBT techniques, in an informative and clean design.",
    #             "caption": "CBT Techniques Tailored for ADHD.",
    #             "voice_over": "Techniques like mindfulness, organization skills, and thought restructuring can be incredibly beneficial for those with ADHD."
    #         },
    #         {
    #             "image_prompt": "Split-screen digital illustration showing an individual before and after using CBT techniques, transitioning from disorganized to organized, and from distracted to focused, in a vibrant style.",
    #             "caption": "Real Success Stories.",
    #             "voice_over": "Meet those who've transformed their lives with CBT. From chaos to clarity, see the difference it makes."
    #         },
    #         {
    #             "image_prompt": "Digital painting of a cozy therapy consultation room with a welcoming atmosphere, subtle contact details visible, in a comforting and realistic style.",
    #             "caption": "Begin Your Journey Today.",
    #             "voice_over": "Interested in exploring how CBT can help with ADHD? Contact a professional to get started on your personalized therapy plan."
    #         }
    #     ]
    # }

    story = {
        "story_title": "The Evolution of Psychology as a Scientific Discipline",
        "story_description": "Explore the fascinating journey of Psychology from its roots in philosophy to its establishment as a scientific discipline. Witness how varying theoretical perspectives and approaches have shaped the field, leading to evolving definitions and methodologies. Delve into the study of human behavior and mental processes, unraveling the complexities of the mind through engaging visuals and informative narration. Discover the unique relationship between psychologists and their subject matter, highlighting the use of cognitive processes to investigate human psychology. Navigate through the blurred boundaries between psychology and other disciplines like physiology, sociology, and anthropology, showcasing the interdisciplinary nature of the field. Learn about the origins of the term 'psychology' and its formal inception in 1879 with Wilhelm Wundt's groundbreaking psychological laboratory. Experience the historical shift towards empirical methods and controlled experimentation, marking a significant departure from its philosophical origins. Immerse yourself in the diverse topics within psychology, aiming to understand human behavior, cognition, emotions, and mental processes through the lens of scientific inquiry and observation.",
        "complete_voice_over": "Explore the fascinating journey of Psychology from its roots in philosophy to its establishment as a scientific discipline. Witness how varying theoretical perspectives and approaches have shaped the field, leading to evolving definitions and methodologies. Delve into the study of human behavior and mental processes, unraveling the complexities of the mind through engaging visuals and informative narration. Discover the unique relationship between psychologists and their subject matter, highlighting the use of cognitive processes to investigate human psychology. Navigate through the blurred boundaries between psychology and other disciplines like physiology, sociology, and anthropology, showcasing the interdisciplinary nature of the field. Learn about the origins of the term 'psychology' and its formal inception in 1879 with Wilhelm Wundt's groundbreaking psychological laboratory. Experience the historical shift towards empirical methods and controlled experimentation, marking a significant departure from its philosophical origins. Immerse yourself in the diverse topics within psychology, aiming to understand human behavior, cognition, emotions, and mental processes through the lens of scientific inquiry and observation.",
        "scenes": [
            {
                "image_prompt": "Image of ancient philosophers discussing the nature of the mind",
                "caption": "Roots in Philosophy",
                "voice_over": "Long ago, ancient philosophers pondered the mysteries of the mind, laying the groundwork for the future field of psychology.",
            },
            {
                "image_prompt": "Image of Wilhelm Wundt conducting experiments in his psychological laboratory",
                "caption": "Birth of Scientific Psychology",
                "voice_over": "In 1879, Wilhelm Wundt established the first psychological laboratory, marking the formal inception of psychology as a scientific discipline.",
            },
            {
                "image_prompt": "Image of modern research methods and technologies used in psychology",
                "caption": "Empirical Methods and Controlled Experimentation",
                "voice_over": "Psychology underwent a historical shift towards empirical methods and controlled experimentation, revolutionizing the study of human behavior and mental processes.",
            },
            {
                "image_prompt": "Image showcasing interdisciplinary collaboration between psychology and other fields",
                "caption": "Interdisciplinary Nature of Psychology",
                "voice_over": "Psychology intertwines with disciplines like physiology, sociology, and anthropology, highlighting its interdisciplinary nature and diverse perspectives on human psychology.",
            },
        ],
    }

    # create a directory inside images folder with the name of story title if it doesn't exist
    if not os.path.exists(f"images/{story['story_title']}"):
        os.mkdir(f"images/{story['story_title']}")
    print(type(story))
    # story = json.dumps(story)
    print(story)
    # raise ValueError("Story JSON is not generated yet.")
    for key in story:
        print(key)
    story_title = story["story_title"]
    story_description = story["story_description"]
    story_voice_over = story["complete_voice_over"]
    scenes = story["scenes"]
    print(story)
        

    # images = []
    clips = []
    # i = 0
    # for scene in scenes:
    #     image_prompt = scene['image_prompt']
    #     caption = scene['caption']
    #     voice_over = scene['voice_over']
    #     # For the last image, add cta of book session with us
    #     if scene == scenes[-1]:
    #         cta = "Book a session"
    #     else :
    #         cta = None
    #     caption_prompt = generate_caption_prompt(image_prompt, caption, cta)
    #     print(caption_prompt)
    #     caption_image = create_main_image(caption_prompt)
    #     save_image_from_url(caption_image, f"images/{story_title}/{i}.jpg")
    #     i = i + 1
    #     images.append(caption_image)
    #     #raise ValueError("Image generation is not done yet.")

    # print(images)

    # clip = ImageSequenceClip(f"images/{story_title}/", durations=[3,3,3,3,3])  # Each image lasts 3 second
    for i in range(5):
        duration = 5000
        fps = 30
        output_filename = os.path.join("clips", f"{story_title}_{i}_video.mp4")
        print(output_filename)
        img = cv2.imread(f"images/{story_title}/{i}.jpg")
        clip_edited = generate_clip(img, duration, fps, output_filename)
        clips.append(clip_edited)
        print(clips)

    # Change voice over to make it therapeutic
    # voice_over = therapeutic_voice_over(story_voice_over)
    audio_path = "sounds/The Evolution of Psychology as a Scientific Discipline_audio.mp3"
    print(audio_path)

    # calculate the duration of the audio
    audio_duration = AudioFileClip(audio_path).duration
    durations_list = [audio_duration / 5] * 5

    # clip = ImageSequenceClip(f"images/{story_title}/", durations=durations_list)
    # clip.write_videofile(f"videos/{story_title}_video.mp4", fps=1)
    # clips = [VideoFileClip(clip) for clip in clips]  # Resize to 720p if necessary
    # final_clip = concatenate_videoclips(clips, method="compose", padding=-1)
    # print(final_clip)
    def resize_frame(frame, width, height):
        return cv2.resize(frame, (width, height))
    # Assuming all clips have the same frame rate and similar properties
    output_filename = "final_video.mp4"
    cap = cv2.VideoCapture(clips[0])
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = 30  # frames per second
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    for clip in clips:
        cap = cv2.VideoCapture(clip)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = resize_frame(frame, frame_width, frame_height)
            out.write(resized_frame)
        cap.release()
    out.release()
    final_video = VideoFileClip(output_filename)
    audio_clip = AudioFileClip(audio_path)
    final_clip = final_video.set_audio(audio_clip)
    final_clip.write_videofile("final_with_audio.mp4")
    print("runnnneddd")
    # Add audio to video using moviepy
    return "final_with_audio.mp4"


if __name__ == "__main__":
    app.run(debug=True)
