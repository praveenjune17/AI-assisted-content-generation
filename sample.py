import os
import yaml
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


def get_authenticated_service(secrets_file, scopes, api_version):
    """
    Authenticates and returns the YouTube API service.

    Args:
        secrets_file (str): Path to the client secrets file for authentication.
        scopes (list): List of API scopes for authentication.
        api_version (str): API version to use for the service.

    Returns:
        googleapiclient.discovery.Resource: YouTube API service object.
    """
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(secrets_file, scopes)
    credentials = flow.run_local_server(port=0)
    return googleapiclient.discovery.build("youtube", api_version, credentials=credentials)


def upload_video(youtube, video_file, title, description, category_id, tags, thumbnail_folder, privacy_status,
                 publish_at):
    """
    Uploads a video to YouTube and sets a custom thumbnail.

    Args:
        youtube (googleapiclient.discovery.Resource): YouTube API service object.
        video_file (str): Path to the video file to upload.
        title (str): Title of the video.
        description (str): Description of the video.
        category_id (str): Category ID for the video.
        tags (list): List of tags for the video.
        thumbnail_folder (str): Path to the custom thumbnail file.
        privacy_status (str): Privacy status of the video (e.g., 'public', 'private', 'unlisted').
        publish_at (str): ISO 8601 timestamp of when to publish the video.
    """
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

    video_insert_request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=googleapiclient.http.MediaFileUpload(video_file, chunksize=-1, resumable=True)
    )

    response_upload = video_insert_request.execute()
    print("Uploaded video ID:", response_upload["id"])

    youtube.thumbnails().set(
        videoId=response_upload["id"],
        media_body=googleapiclient.http.MediaFileUpload(thumbnail_folder)
    ).execute()

    print("Custom thumbnail set for video ID:", response_upload["id"])


def process_video(video_path, file, seconds_per_frame=2):
    """
    Extracts frames and audio from a video file.

    Args:
        video_path (str): Path to the video directory.
        file (str): Video filename.
        seconds_per_frame (int, optional): Number of seconds to skip between each frame extraction. Defaults to 2.

    Returns:
        tuple: A tuple containing a list of base64-encoded frames and the path to the extracted audio file.
    """
    base64Frames = []
    video = cv2.VideoCapture(video_path + '/' + file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    audio_file = "audio.mp3"
    clip = VideoFileClip(video_path + file)
    clip.audio.write_audiofile(video_path + audio_file, bitrate="32k")
    clip.audio.close()
    clip.close()
    audio_path = video_path + audio_file
    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path


def get_thumbnail(thumbnail_folder, img_gen_model, thumbnail_prompt, thumbnail_resolution=(1280, 720)):
    """
    Generates a thumbnail using the specified image generation model and saves it to a file.

    Args:
        thumbnail_folder (str): Path where the generated thumbnail will be saved.
        img_gen_model (str): Image generation model to use.
        thumbnail_prompt (str): Prompt for generating the thumbnail image.
        thumbnail_resolution (tuple, optional): Resolution to resize the thumbnail to. Defaults to (1280, 720).
    """
    response = client.images.generate(
        model=img_gen_model,
        prompt=thumbnail_prompt,
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    thumbnail_file_path = os.path.join(thumbnail_folder, 'thumbnail.jpg')
    with open(thumbnail_file_path, 'wb') as handler:
        handler.write(img_data)

    image = Image.open(thumbnail_file_path)
    resized_image = image.resize(thumbnail_resolution)
    resized_image.save(thumbnail_file_path)
    return thumbnail_file_path


def get_publish_time(time_zone, publish_yr, publish_mth, publish_day, publish_hr, publish_min):
    """
    Converts the given time details into a UTC ISO 8601 timestamp for YouTube video scheduling.

    Args:
        time_zone (str): Time zone for the publish time (e.g., 'Asia/Kolkata').
        publish_yr (int): Year of the scheduled publish time.
        publish_mth (int): Month of the scheduled publish time.
        publish_day (int): Day of the scheduled publish time.
        publish_hr (int): Hour of the scheduled publish time.
        publish_min (int): Minute of the scheduled publish time.

    Returns:
        str: ISO 8601 formatted UTC publish time.
    """
    local_tz = pytz.timezone(time_zone)
    scheduled_time = local_tz.localize(
        datetime.datetime(publish_yr, publish_mth, publish_day, publish_hr, publish_min, 0))
    publish_time = scheduled_time.astimezone(pytz.utc).isoformat()
    return publish_time


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI()
    with open('/Users/praveenkumarchandrasekaran/PycharmProjects/workflow/pr_agents/conf/config.yml') as f:
        config = yaml.safe_load(f)

    model = config['llm']
    video_file_folder = config['video_file_folder']
    video_file = config['video_file']
    thumbnail_folder = config['thumbnail_folder']
    thumbnail_resolution = config['thumbnail_resolution']
    img_gen_model = config['img_gen_model']
    video_privacy_status = config['video_privacy_status']
    secrets_file = config['secrets_file']
    scopes = config['scopes']
    api_version = config['api_version']
    category_id = config['category_id']
    time_zone = config['time_zone']
    asr_model = config['asr_model']
    language = config['language']
    llm_prompt = config['system_prompt'].format(language)
    date_to_publish = config['date_to_publish']

    publish_yr = date_to_publish.get('publish_yr', datetime.datetime.today().year)
    publish_mth = date_to_publish.get('publish_mth', datetime.datetime.today().month)
    publish_day = date_to_publish.get('publish_day', datetime.datetime.today().day)
    publish_hr = date_to_publish.get('publish_hr', datetime.datetime.today().hour + 1)
    publish_min = date_to_publish.get('publish_min', datetime.datetime.today().minute)
    date_check_condition = datetime.datetime.today() <= datetime.datetime(publish_yr, publish_mth, publish_day,
                                                                          publish_hr, publish_min)
    assert date_check_condition, 'The publish date needs to be today or any date in the future'
    video_path = os.path.join(video_file_folder, video_file)
    base64Frames, audio_path = process_video(video_file_folder, file=video_file, seconds_per_frame=1)

    transcription = client.audio.transcriptions.create(
        model=asr_model,
        file=open(audio_path, "rb"),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{llm_prompt}"},
            {"role": "user", "content": [
                "These are the frames from the video.",
                *map(lambda x: {"type": "image_url",
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}},
                     base64Frames),
                {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
            ]}
        ],
        temperature=0,
    )

    output = json.loads(response.json())
    output = json.loads(output['choices'][0]['message']['content'].split('```json')[1].split('```')[0])
    title = output['title']
    description = output['description']
    thumbnail_response = output['thumbnail']
    tags = output['keywords']

    thumbnail_idea = thumbnail_response['idea']
    thumbnail_style = thumbnail_response['style']
    thumbnail_prompt = f'Create a thumbnail of {thumbnail_idea}, make sure it is styled as a {thumbnail_style}'
    print(thumbnail_prompt)
    thumbnail_file = get_thumbnail(thumbnail_folder, img_gen_model, thumbnail_prompt)
    print(f'Thumbnail file path-> {thumbnail_file}')
    print()
    youtube = get_authenticated_service(secrets_file, scopes, api_version)
    publish_at = get_publish_time(time_zone, publish_yr, publish_mth, publish_day, publish_hr, publish_min)
    print(f'The video will be published at {publish_at}')
    # Uncomment to upload video after testing
    print('Uploading the content to youtube')
    upload_video(youtube, video_path, title, description, category_id, tags,
                 thumbnail_file, video_privacy_status, publish_at)
