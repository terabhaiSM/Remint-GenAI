"""
!pip install --quiet ipython-autotime
import os

mp4videoURL = ""
videofilename = mp4videoURL.split('/')[-1]
print (videofilename)

import urllib.request
urllib.request.urlretrieve(mp4videoURL, videofilename)

"""## Step2: Extract audio (mp3) from the video file"""

!pip install ffmpeg-python==0.2.0

import ffmpeg

audiofilename = videofilename.replace(".mp4",'.mp3')

# Create the ffmpeg input stream
input_stream = ffmpeg.input(videofilename)

# Extract the audio stream from the input stream
audio = input_stream.audio

# Save the audio stream as an MP3 file
output_stream = ffmpeg.output(audio, audiofilename)

# Overwrite output file if it already exists
output_stream = ffmpeg.overwrite_output(output_stream)

ffmpeg.run(output_stream)

print (audiofilename)

from IPython.display import Audio

Audio(audiofilename)

"""## Step3: Get word-level transcription (speech-to-text) from audio using OpenAI's Whisper"""

!pip install git+https://github.com/openai/whisper.git@248b6cb124225dd263bb9bd32d060b6517e067f8

import whisper
                                          #might take some time (approx 3- 5min depending on audio length)
model = whisper.load_model("medium")
result = model.transcribe(audiofilename,word_timestamps=True)
print(result)

# total text
print (result['text'])

# each subtitle segment
for each in result['segments']:
  print (each)

wordlevel_info = []

for each in result['segments']:
  words = each['words']
  for word in words:
    # print (word['word'], "  ",word['start']," - ",word['end'])
    wordlevel_info.append({'word':word['word'].strip(),'start':word['start'],'end':word['end']})

wordlevel_info

"""## Step4: Store word-level timestamps into JSON file"""

import json
with open('data.json', 'w') as f:
    json.dump(wordlevel_info, f,indent=4)

"""## Step5: Correct and save JSON file if there are transcription errors or if you need to merge any words into phrases. Load the new edited JSON file with new word-level timestamps"""

import json

with open('data.json', 'r') as f:
    wordlevel_info_modified = json.load(f)

wordlevel_info_modified

"""## Step6: Convert word-level timestamps JSON to line-level timestamps JSON. We want to display one line at a time and highlight words in it as they are spoken so we do this transformation."""

def split_text_into_lines(data):

    MaxChars = 80
    #maxduration in seconds
    MaxDuration = 3.0
    #Split if nothing is spoken (gap) for these many seconds
    MaxGap = 1.5

    subtitles = []
    line = []
    line_duration = 0
    line_chars = 0


    for idx,word_data in enumerate(data):
        word = word_data["word"]
        start = word_data["start"]
        end = word_data["end"]

        line.append(word_data)
        line_duration += end - start

        temp = " ".join(item["word"] for item in line)


        # Check if adding a new word exceeds the maximum character count or duration
        new_line_chars = len(temp)

        duration_exceeded = line_duration > MaxDuration
        chars_exceeded = new_line_chars > MaxChars
        if idx>0:
          gap = word_data['start'] - data[idx-1]['end']
          # print (word,start,end,gap)
          maxgap_exceeded = gap > MaxGap
        else:
          maxgap_exceeded = False


        if duration_exceeded or chars_exceeded or maxgap_exceeded:
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line
                }
                subtitles.append(subtitle_line)
                line = []
                line_duration = 0
                line_chars = 0


    if line:
        subtitle_line = {
            "word": " ".join(item["word"] for item in line),
            "start": line[0]["start"],
            "end": line[-1]["end"],
            "textcontents": line
        }
        subtitles.append(subtitle_line)

    return subtitles

linelevel_subtitles = split_text_into_lines(wordlevel_info_modified)
print (linelevel_subtitles)

for line in linelevel_subtitles:
  json_str = json.dumps(line, indent=4)
  print(json_str)

"""## Step7: Use Moviepy to create an audiogram with word-level highlights as they are spoken"""

!pip install moviepy==2.0.0.dev2
!pip install imageio==2.25.1

!apt install imagemagick

!cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml

from moviepy.editor import TextClip, CompositeVideoClip, ColorClip
import numpy as np


def create_caption(textJSON, framesize,font = "Helvetica-Bold",fontsize=80, color='white', bgcolor='blue'):
    wordcount = len(textJSON['textcontents'])
    full_duration = textJSON['end']-textJSON['start']

    word_clips = []
    xy_textclips_positions =[]

    x_pos = 0
    y_pos = 0
    # max_height = 0
    frame_width = framesize[0]
    frame_height = framesize[1]
    x_buffer = frame_width*1/10
    y_buffer = frame_height*1/5

    space_width = ""
    space_height = ""

    for index,wordJSON in enumerate(textJSON['textcontents']):
      duration = wordJSON['end']-wordJSON['start']
      word_clip = TextClip(wordJSON['word'], font = font,fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
      word_clip_space = TextClip(" ", font = font,fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
      word_width, word_height = word_clip.size
      space_width,space_height = word_clip_space.size
      if x_pos + word_width+ space_width > frame_width-2*x_buffer:
            # Move to the next line
            x_pos = 0
            y_pos = y_pos+ word_height+40

            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos+x_buffer,
                "y_pos": y_pos+y_buffer,
                "width" : word_width,
                "height" : word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos+x_buffer, y_pos+y_buffer))
            word_clip_space = word_clip_space.set_position((x_pos+ word_width +x_buffer, y_pos+y_buffer))
            x_pos = word_width + space_width
      else:
            # Store info of each word_clip created
            xy_textclips_positions.append({
                "x_pos":x_pos+x_buffer,
                "y_pos": y_pos+y_buffer,
                "width" : word_width,
                "height" : word_height,
                "word": wordJSON['word'],
                "start": wordJSON['start'],
                "end": wordJSON['end'],
                "duration": duration
            })

            word_clip = word_clip.set_position((x_pos+x_buffer, y_pos+y_buffer))
            word_clip_space = word_clip_space.set_position((x_pos+ word_width+ x_buffer, y_pos+y_buffer))

            x_pos = x_pos + word_width+ space_width


      word_clips.append(word_clip)
      word_clips.append(word_clip_space)


    for highlight_word in xy_textclips_positions:

      word_clip_highlight = TextClip(highlight_word['word'], font = font,fontsize=fontsize, color=color,bg_color = bgcolor).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
      word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
      word_clips.append(word_clip_highlight)

    return word_clips

from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips,VideoFileClip, ColorClip
frame_size = (1080,1080)

all_linelevel_splits=[]

for line in linelevel_subtitles:
  out = create_caption(line,frame_size)
  all_linelevel_splits.extend(out)


# Load the input video
input_video = VideoFileClip(videofilename)
# Get the duration of the input video
input_video_duration = input_video.duration
# Create a color clip with the given frame size, color, and duration
background_clip = ColorClip(size=frame_size, color=(0, 0, 0)).set_duration(input_video_duration)

# If you want to overlay this on the original video uncomment this and also change frame_size, font size and color accordingly.
# final_video = CompositeVideoClip([input_video] + all_linelevel_splits)

final_video = CompositeVideoClip([background_clip] + all_linelevel_splits)

# Set the audio of the final video to be the same as the input video
final_video = final_video.set_audio(input_video.audio)

# Save the final clip as a video file with the audio included
final_video.write_videofile("output.mp4", fps=24, codec="libx264", audio_codec="aac")

"""## Step8: Visualize the final Descript like audiogram video"""

from IPython.display import HTML
from base64 import b64encode
mp4 = open("output.mp4",'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)