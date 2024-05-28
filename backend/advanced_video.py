import os
import numpy as np
import dlib
import moviepy.editor as mp
from moviepy.editor import VideoClip, TextClip, CompositeVideoClip, ImageClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
import math

# Facial landmark predictor path
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

def add_caption_with_shape(image_path, caption, output_path, shape='rectangle'):
    # Load the image
    img = Image.open(image_path)
    
    # Define the text properties for the caption
    font_path = "/Users/sonalidixit/Library/Fonts/DejaVuSans-Bold.ttf"
    caption_font = ImageFont.truetype(font_path, 40)
    
    # Calculate text width and height using getbbox for the caption
    text_bbox = caption_font.getbbox(caption)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Define the shape properties for the caption
    shape_width = text_width + 40  # padding
    shape_height = text_height + 40
    shape_x = (img.width - shape_width) // 2
    shape_y = 10  # 10 pixels from the top
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Determine the average brightness to choose the shape color
    brightness = sum(img.convert("L").getdata()) / (img.width * img.height)
    shape_color = "black" if brightness > 128 else "white"  # simple thresholding at mid-point
    text_color = "white" if shape_color == "black" else "black"
    
    # Draw the shape on the image for the caption
    if shape == 'rectangle':
        draw.rectangle([shape_x, shape_y, shape_x + shape_width, shape_y + shape_height], fill=shape_color)
    elif shape == 'ellipse':
        draw.ellipse([shape_x, shape_y, shape_x + shape_width, shape_y + shape_height], fill=shape_color)
    elif shape == 'polygon':
        # Example: Create a hexagon
        points = [(shape_x + shape_width/2, shape_y),
                  (shape_x + shape_width, shape_y + shape_height/3),
                  (shape_x + shape_width, shape_y + 2*shape_height/3),
                  (shape_x + shape_width/2, shape_y + shape_height),
                  (shape_x, shape_y + 2*shape_height/3),
                  (shape_x, shape_y + shape_height/3)]
        draw.polygon(points, fill=shape_color)
    # Add more shapes as needed
    
    # Add text over the shape for the caption
    text_x = shape_x + 20  # 20 pixels padding
    text_y = shape_y + 20
    draw.text((text_x, text_y), caption, font=caption_font, fill=text_color)
    
    # Save the modified image
    img.save(output_path)

def crop_avatar_face(avatar_path):
    # Load avatar image
    avatar = Image.open(avatar_path).convert('RGBA')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    # Convert avatar to numpy array
    avatar_array = np.array(avatar.convert('RGB'))
    
    # Detect face
    dets = detector(avatar_array, 1)
    if len(dets) > 0:
        # Get the first detected face
        shape = predictor(avatar_array, dets[0])
        # Get the bounding box of the face
        x_coords = [shape.part(i).x for i in range(17)]
        y_coords = [shape.part(i).y for i in range(17)]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        # Crop the face
        face = avatar.crop((x_min, y_min, x_max, y_max))
        return face
    return avatar  # Return original avatar if no face detected

def create_typing_text_clip(narration, font_path, fontsize, color, audio_duration, position, video_size):
    sentences = narration.split('. ')  # Split narration into sentences
    print(len(sentences))
    print(audio_duration)
    duration_of_sentence = math.ceil(audio_duration/len(sentences))
    print(duration_of_sentence)
    clips = []
    
    for i, sentence in enumerate(sentences):
        if sentence:  # Skip empty sentences
            # Find duration of sentence
            duration_of_sentence = math.ceil(audio_duration/len(sentences))
            sentence = sentence.strip() + '.'  # Add period back to the sentence
            text_clip = TextClip(sentence, fontsize=fontsize, font=font_path, color=color, method='caption', align='South', size=video_size)
            text_clip = text_clip.set_duration(duration_of_sentence).set_position(position)
            text_clip = text_clip.crossfadein(0.5).crossfadeout(0.5)  # Smooth transition
            text_clip = text_clip.set_start(i * duration_of_sentence)
            clips.append(text_clip)
    
    return CompositeVideoClip(clips)

def create_avatar_mouth_movement(avatar_path, narration_path, video_size):
    # Load and crop avatar image
    avatar = crop_avatar_face(avatar_path).convert('RGBA')
    avatar = avatar.resize((video_size[1] // 3, video_size[1] // 3), Image.LANCZOS)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    # Load the audio narration
    audio_clip = AudioFileClip(narration_path)
    
    # Define mouth coordinates
    def get_mouth_shape(image):
        dets = detector(image, 1)
        if len(dets) > 0:
            shape = predictor(image, dets[0])
            mouth = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)])
            return mouth
        else:
            return None

    # Simulate mouth movement based on the narration's audio levels
    def make_frame(t):
        frame = avatar.copy()
        draw = ImageDraw.Draw(frame)
        # Get the corresponding frame time in the audio
        audio_time = audio_clip.to_soundarray(t)
        if np.mean(audio_time) > 0.01:  # Lowered threshold for mouth opening
            mouth = get_mouth_shape(np.array(frame.convert('RGB')))
            if mouth is not None:
                hull = ConvexHull(mouth)
                draw.polygon([tuple(mouth[v]) for v in hull.vertices], fill=(0, 0, 0, 255))  # Simple black mouth
                print(f"Drawing mouth at time {t}: {hull.vertices}")  # Debugging print statement
            else:
                print(f"No mouth detected at time {t}")  # Debugging print statement
        else:
            print(f"Audio below threshold at time {t}: {np.mean(audio_time)}")  # Debugging print statement
        return np.array(frame.convert('RGB'))

    return VideoClip(make_frame, duration=audio_clip.duration).set_position(('left', 'top')).resize(height=video_size[1] // 3)

def create_video_clip_with_narration(image_path, narration, output_path, avatar_path=None, narration_path=None):
    # Add caption to the image
    temp_image_path = os.path.join(output_path, "0_temp.jpg")
    add_caption_with_shape(image_path, "Kalin bhaiya bol rahe h to risk to hoga", temp_image_path)
    
    
    audio_duration  = AudioFileClip(narration_path).duration
    
    # Load the image to get the size
    img = Image.open(temp_image_path)
    video_size = img.size  # Use image size for video size
    
    # Create the narration TextClip with typing effect
    font_path = "/Users/sonalidixit/Library/Fonts/DejaVuSans-Bold.ttf"
    text_clip = create_typing_text_clip(narration, font_path, 25, 'white', audio_duration, ('center', 'bottom'), video_size)
    
    # Create a background shape for the subtitles
    def add_shape_background(get_frame, t):
        frame = get_frame(t)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        shape_x = 0
        shape_y = img.height - 100  # Adjust as needed
        shape_width = img.width
        shape_height = 100
        draw.rectangle([shape_x, shape_y, shape_x + shape_width, shape_y + shape_height], fill='black')
        return np.array(img)
    
    # Load the modified image as VideoClip
    image_clip = VideoClip(lambda t: np.array(Image.open(temp_image_path).convert('RGB')), duration=text_clip.duration)
    image_clip = image_clip.fl(add_shape_background)
    
    clips = [image_clip, text_clip]
    
    # Add avatar with mouth movement if provided
    if avatar_path and narration_path:
        #avatar_clip = create_avatar_mouth_movement(avatar_path, narration_path, video_size)
        #clips.append(avatar_clip)
        pass
    
    # Combine the image and text clips
    final_clip = CompositeVideoClip(clips, size=video_size)

    # Add audio narration
    audio_clip = AudioFileClip(narration_path)
    final_clip = final_clip.set_audio(audio_clip)

    
    # Write the final clip to a video file
    final_clip.write_videofile(os.path.join(output_path, "video_clip.mp4"), codec='libx264', fps=24)
    
    # Remove temporary image file
    os.remove(temp_image_path)

# Example usage:
create_video_clip_with_narration(
    '/Users/sonalidixit/Desktop/Operations tool/Version-1/image_tool/backend/images/Decoding Property Risks in India: Your Ultimate Guide to Safe Investment/0.jpg', 
    'Risk is the potential of losing something of value. Values can be gained or lost when taking risk resulting from a given action or inaction, foreseen or unforeseen.', 
    '/Users/sonalidixit/Desktop/Operations tool/Version-1/image_tool/backend/images/Decoding Property Risks in India: Your Ultimate Guide to Safe Investment/', 
    avatar_path='/Users/sonalidixit/Desktop/Operations tool/Version-1/image_tool/backend/images/Decoding Property Risks in India: Your Ultimate Guide to Safe Investment/Avatar.png', 
    narration_path='/Users/sonalidixit/Desktop/Operations tool/Version-1/image_tool/backend/images/Decoding Property Risks in India: Your Ultimate Guide to Safe Investment/output.mp3'
)


# Face detection and cropping
