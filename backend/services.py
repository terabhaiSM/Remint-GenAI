from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip, TextClip, CompositeVideoClip
import numpy as np
import os

def add_caption_with_shape(image_path, caption, output_path):
    # Load the image
    img = Image.open(image_path)
    
    # Define the text properties for the caption
    font_path = "/Users/sonalidixit/Library/Fonts/DejaVuSans-Bold.ttf"
    caption_font = ImageFont.truetype(font_path, 30)
    
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
    draw.rectangle([shape_x, shape_y, shape_x + shape_width, shape_y + shape_height], fill=shape_color)
    
    # Add text over the shape for the caption
    text_x = shape_x + 20  # 20 pixels padding
    text_y = shape_y + 20
    draw.text((text_x, text_y), caption, font=caption_font, fill=text_color)
    
    # Save the modified image
    img.save(output_path)

def create_typing_text_clip(narration, font_path, fontsize, color, duration_per_sentence, position, video_size):
    sentences = narration.split('. ')  # Split narration into sentences
    clips = []
    for i, sentence in enumerate(sentences):
        if sentence:  # Skip empty sentences
            sentence = sentence.strip() + '.'  # Add period back to the sentence
            text_clip = TextClip(sentence, fontsize=fontsize, font=font_path, color=color, method='caption', align='South', size=video_size)
            text_clip = text_clip.set_duration(duration_per_sentence).set_position(position)
            text_clip = text_clip.crossfadein(0.5).crossfadeout(0.5)  # Smooth transition
            text_clip = text_clip.set_start(i * duration_per_sentence)
            clips.append(text_clip)
    
    return CompositeVideoClip(clips)

def create_video_clip_with_narration(image_path, narration, output_path):
    # Add caption to the image
    temp_image_path = output_path + "0_temp.jpg"
    add_caption_with_shape(image_path, "Kalin bhaiya bol rahe h to risk to hoga", temp_image_path)
    
    # Duration for each sentence
    duration_per_sentence = 3  # 3 seconds per sentence (adjust as needed)
    
    # Load the image to get the size
    img = Image.open(temp_image_path)
    video_size = img.size  # Use image size for video size
    
    # Create the narration TextClip with typing effect
    font_path = "/Users/sonalidixit/Library/Fonts/DejaVuSans-Bold.ttf"
    text_clip = create_typing_text_clip(narration, font_path, 25, 'white', duration_per_sentence, ('center', 'bottom'), video_size)
    
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

    # Combine the image and text clips
    final_clip = CompositeVideoClip([image_clip, text_clip])
    
    # Write the final clip to a video file
    final_clip.write_videofile(output_path + "video_clip.mp4", codec='libx264', fps=24)
    
    # Remove temporary image file
    os.remove(temp_image_path)

# Example usage:
create_video_clip_with_narration('/Users/sonalidixit/Desktop/Operations tool/Version-1/image_tool/backend/images/Decoding Property Risks in India: Your Ultimate Guide to Safe Investment/0.jpg', 'What is the risk in property investment? Let me explain. Risk is the potential of losing something of value. Values can be gained or lost when taking risk resulting from a given action or inaction, foreseen or unforeseen. Risk can also be defined as the intentional interaction with uncertainty. Uncertainty is a potential, unpredictable, and uncontrollable outcome; risk is an aspect of action taken in spite of uncertainty. Risk perception is the subjective judgment people make about the severity and probability of a risk, and may vary person to person. Any investment in property carries some risk, but with the right knowledge and guidance, you can make informed decisions to minimize these risks. Let me guide you through the risks involved in property investment.', '/Users/sonalidixit/Desktop/Operations tool/Version-1/image_tool/backend/images/Decoding Property Risks in India: Your Ultimate Guide to Safe Investment/')
