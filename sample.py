import os
from openai import OpenAI
import requests
import json
import cv2
import pytz
import datetime
from moviepy.editor import VideoFileClip
import base64
from PIL import Image
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.http
from dotenv import load_dotenv

# Replace with your OAuth 2.0 credentials file
CLIENT_SECRETS_FILE = "/Users/praveenkumarchandrasekaran/Downloads/client_secrets_v2.json"
# Scopes for uploading video and managing it
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

def get_authenticated_service():
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)

    # Use run_local_server() instead of run_console()
    credentials = flow.run_local_server(port=0)
    return googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

def upload_video(youtube, video_file, title,
                 description, category_id, tags,
                 thumbnail_file, privacy_status,
                 publish_at):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id
        },
        "status": {
            "privacyStatus": privacy_status,
            "publishAt": publish_at

        }
    }

    # Upload the video
    video_insert_request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=googleapiclient.http.MediaFileUpload(video_file, chunksize=-1, resumable=True)
    )

    response_upload = video_insert_request.execute()
    print("Uploaded video ID:", response_upload["id"])

    # Upload the custom thumbnail
    youtube.thumbnails().set(
        videoId=response_upload["id"],
        media_body=googleapiclient.http.MediaFileUpload(thumbnail_file)
    ).execute()

    print("Custom thumbnail set for video ID:", response_upload["id"])

def process_video(video_path, file, seconds_per_frame=2):
    base64Frames = []
    #base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path+'/'+file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_file = "audio.mp3"
    clip = VideoFileClip(video_path+file)
    clip.audio.write_audiofile(video_path+audio_file, bitrate="32k")
    clip.audio.close()
    clip.close()
    audio_path = video_path + audio_file
    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

def get_thumbnail(thumbnail_file, thumbnail_prompt):
    '''

    :return:
    '''

    response = client.images.generate(
        model="dall-e-3",
        prompt=thumbnail_prompt,
        size="1792x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    with open(thumbnail_file, 'wb') as handler:
        handler.write(img_data)


    # Open the image
    image = Image.open(thumbnail_file)
    # Resize the image
    resized_image = image.resize((1280, 720))
    # Save the resized image
    resized_image.save(thumbnail_file)


if __name__ == "__main__":

    load_dotenv()
    MODEL = "gpt-4o-mini"
    client = OpenAI(
            #api_key=os.environ.get("OPENAI_API_KEY", "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")
    )
    video_file_folder = "/Users/praveenkumarchandrasekaran/Movies/Wondershare Filmora Mac/Recorded/udemy_lectures/"
    video_file = "VID_20240806_145608.mp4"
    thumbnail_file = "/Users/praveenkumarchandrasekaran/Downloads/image_name_v2.jpg"
    video_privacy_status = 'private' # Can be "private", "public", or "unlisted"
    publish_yr  = datetime.datetime.today().year
    publish_mth = datetime.datetime.today().month
    publish_day = datetime.datetime.today().day
    publish_hr  = 18
    publish_min = 0

    # Replace with your local timezone
    local_tz = pytz.timezone("asia/kolkata")
    # Define the schedule date and time in local timezone
    scheduled_time = local_tz.localize(datetime.datetime(publish_yr,
                                                         publish_mth,
                                                         publish_day,
                                                         publish_hr,
                                                         publish_min,
                                                         0)
                                       )

    # Convert to UTC
    publish_at = scheduled_time.astimezone(pytz.utc).isoformat()

    # Transcribe the audio
    # Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
    base64Frames, audio_path = process_video(video_file_folder,
                                             file=video_file,
                                             seconds_per_frame=1)
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": """
        Using the provided information create a potential high
        click-through-rate youtube title and a catchy description explaining about
        the video with some trending hashtags.
        Make sure both are generated in Tamil (with a mix of english here and there).
        Generate a relevant thumbnail to be used for this video and make sure it is 
        cute and friendly and also add some keywords (with hashtags relating to AI, TAMIL)
        related to this content. Return the youtube title, description, thumbnail idea,
        and the keywords structured in JSON {title, description, thumbnail, keywords}
        """},
            {"role": "user", "content": [
                "These are the frames from the video.",
                *map(lambda x: {"type": "image_url",
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
            ],
             }
        ],
        temperature=0,
    )
    output = json.loads(response.json())
    output = json.loads(output['choices'][0]['message']['content'].split('```json')[1].split('```')[0])
    title = output['title']
    description = output['description']
    thumbnail_prompt = output['thumbnail']
    tags = output['keywords']
    category_id = "27"  # e.g., "22" for People & Blogs
    get_thumbnail(thumbnail_file, thumbnail_prompt)

    # Authenticate to the YouTube API
    youtube = get_authenticated_service()
    # Upload the video and set the thumbnail
    video_path = os.path.join(video_file_folder, video_file)
    upload_video(youtube, video_path, title, description, category_id,
                 tags, thumbnail_file, video_privacy_status, publish_at)
