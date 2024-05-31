import cv2
import numpy as np

# Load the image
img = cv2.imread('/Users/mradulsingh/Reminted/backend/images/Decoding Property Risks in India: Your Ultimate Guide to Safe Investment/0.jpg')

# Get the dimensions of the image
height, width, channels = img.shape

# Define the duration of the clip in milliseconds (5 seconds = 5000 milliseconds)
duration = 5000

# Define the frame rate (30 frames per second)
fps = 30

# Calculate the number of frames
n_frames = int(duration * fps / 1000)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Define the zoom factor for the start and end
zoom_factor_start = 1.0
zoom_factor_end = 0.5

# Define the zoom center for the middle part
zoom_center_x = int(width / 2)
zoom_center_y = int(height / 2)

# Loop through the frames
for i in range(n_frames):
    # Calculate the current zoom factor
    if i < n_frames // 3:  # Zoom in for the first 3 seconds
        zoom_factor = zoom_factor_start + (1.0 - zoom_factor_start) * i / (n_frames // 3)
    elif i < 2 * n_frames // 3:  # Stay zoomed in for the middle part
        zoom_factor = 1.0
    else:  # Zoom out for the last 1 second
        zoom_factor = 1.0 - (1.0 - zoom_factor_end) * (i - 2 * n_frames // 3) / (n_frames // 3)

    # Apply the zoom effect
    if zoom_factor != 1.0:
        M = np.float32([[zoom_factor, 0, (1 - zoom_factor) * zoom_center_x],
                        [0, zoom_factor, (1 - zoom_factor) * zoom_center_y]])
        img_zoomed = cv2.warpAffine(img, M, (width, height))
    else:
        img_zoomed = img.copy()

    # Apply the fade out effect for the last frame
    if i == n_frames - 1:
        img_zoomed = np.uint8(img_zoomed * 0.0)

    # Write the frame to the video
    out.write(img_zoomed)

# Release the VideoWriter object
out.release()