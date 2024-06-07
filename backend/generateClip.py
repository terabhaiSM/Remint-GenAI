import cv2
import numpy as np
import urllib.request
from IPython.display import Video

def generate_clip(img, duration, fps, output_filename):
    # Get the dimensions of the image
    height, width, channels = img.shape

    # Calculate the number of frames
    n_frames = int(duration * fps / 1000)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Define the zoom factors
    zoom_factor_start = 1.2
    zoom_factor_end = 1.0
    zoom_factor_initial = 1.0

    # Define the fade out effect
    fade_out_duration = 30  # frames (1 second at 30 fps)

    # # Define subtitles and their durations (in frames)
    # subtitles = [
    #     ("This is the first subtitle", int(n_frames / 5)),
    #     ("Here's the second subtitle", int(n_frames / 5)),
    #     ("Another subtitle appears", int(n_frames / 5)),
    #     ("More text to read here", int(n_frames / 5)),
    #     ("Final subtitle!", int(n_frames / 5))
    # ]

    # # Caption properties
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 1  # Increased font scale for larger text
    # font_color = (255, 255, 255)  # White color
    # thickness = 3
    # caption_position_y = height - 100  # Higher position from the bottom
    # background_color = (0, 0, 0)  # Black background for text

    # Loop through the frames
    for i in range(n_frames):
        # Calculate the current zoom factor
        if i < n_frames // 2:  # Zoom in for the first half
            zoom_factor = zoom_factor_initial + (zoom_factor_start - zoom_factor_initial) * (i / (n_frames // 2))
        else:  # Zoom out for the second half
            zoom_factor = zoom_factor_start - (zoom_factor_start - zoom_factor_end) * ((i - n_frames // 2) / (n_frames // 2))

        # Apply the zoom effect
        center_x, center_y = width // 2, height // 2
        M = np.float32([[zoom_factor, 0, center_x * (1 - zoom_factor)],
                        [0, zoom_factor, center_y * (1 - zoom_factor)]])
        img_zoomed = cv2.warpAffine(img, M, (width, height))

        # Apply the fade-out effect
        if i >= n_frames - fade_out_duration:
            alpha = 1.0 - (i - (n_frames - fade_out_duration)) / fade_out_duration
            img_zoomed = cv2.addWeighted(img_zoomed, alpha, img_zoomed, 0, 0)

        # # Determine which subtitle to show
        # subtitle = ""
        # subtitle_start_frame = 0
        # for text, duration in subtitles:
        #     subtitle_end_frame = subtitle_start_frame + duration
        #     if subtitle_start_frame <= i < subtitle_end_frame:
        #         subtitle = text
        #         break
        #     subtitle_start_frame = subtitle_end_frame

        # # Add subtitle text to the image
        # if subtitle:
        #     text_size = cv2.getTextSize(subtitle, font, font_scale, thickness)[0]
        #     text_x = (width - text_size[0]) // 2
        #     text_y = caption_position_y
        #     # Draw background rectangle
        #     cv2.rectangle(img_zoomed, (text_x - 10, text_y - text_size[1] - 10),
        #                   (text_x + text_size[0] + 10, text_y + 10), background_color, -1)
        #     # Draw text
        #     img_zoomed = cv2.putText(img_zoomed, subtitle, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Write the frame to the video
        out.write(img_zoomed)

    # Release the VideoWriter object
    out.release()

    return output_filename

# Example usage
img = cv2.imread("/Users/mradulsingh/Reminted/backend/images/The Evolution of Psychology as a Scientific Discipline/2.jpg")
video_file = generate_clip(img, 5000, 30, "output.mp4")
# Video(video_file)
