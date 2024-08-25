import os
from flask import Flask, request, render_template, send_file
from dotenv import load_dotenv
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from groq import Groq
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline
import requests
import torch
from torch import autocast
import numpy as np

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

app = Flask(__name__)

def generate_story(query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Write a short story about {query} in 5-6 sentences"
            }
        ],
        model="llama3-8b-8192"
    )
    story = chat_completion.choices[0].message.content
    return story


import re

def generate_images(story):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "RunwayML/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)

    sentences = re.split(r'[.!?]', story) 
    images = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()  
        if sentence:
            prompt = construct_prompt(sentence, story, i)
            image = pipe(prompt, guidance_scale=10, num_inference_steps=50).images[0]
            images.append(image)

    return images


def construct_prompt(sentence, story, sentence_index):
    
    
    context = get_context(story, sentence_index)

   
    prompt = f"An illustration depicting {context}: {sentence}"
    return prompt

def get_context(story, sentence_index, window_size=2):
   
    sentences = re.split(r'[.!?]', story)
    start_idx = max(0, sentence_index - window_size)
    end_idx = min(len(sentences), sentence_index + window_size + 1)
    context = ' '.join(sentences[start_idx:end_idx])
    return context

def text_to_speech(story, audio_file):
    tts = gTTS(text=story, lang='en')
    tts.save(audio_file)

def create_video_with_audio(images, audio_file, video_file):
    clips = []
    duration_per_image = 5  
    for img in images:
        img_np = np.array(img) 
        clip = ImageClip(img_np).set_duration(duration_per_image)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_file)
    video = video.set_audio(audio)
    video.write_videofile(video_file, fps=24)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']

        story = generate_story(query)

        
        images = generate_images(story)

        
        audio_file = "story.mp3"
        text_to_speech(story, audio_file)

       
        video_file = "story.mp4"
        create_video_with_audio(images, audio_file, video_file)

        return render_template('story.html', story=story)

    return render_template('index.html')

@app.route('/video/<path:path>')
def send_video(path):
    return send_file(path)

if __name__ == '__main__':
    app.run(debug=True)