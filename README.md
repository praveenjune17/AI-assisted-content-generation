Here's a sample `README.md` file for the provided code:

---

# YouTube Video Uploader with OpenAI and Google API Integration

This repository contains a Python script designed to automate the process of uploading videos to YouTube. It leverages OpenAI's API to generate descriptions, thumbnails, and titles based on video content, and Google's YouTube API to publish the video with a custom thumbnail.

## Features

- **Video Processing**: Extracts frames and audio from a video file.
- **OpenAI Integration**: Uses OpenAI's LLM for transcription and generates titles, descriptions, and thumbnail prompts.
- **Thumbnail Generation**: Automatically generates and resizes thumbnails based on prompts.
- **Scheduled YouTube Upload**: Authenticates and uploads videos to YouTube with customizable scheduling.

## Prerequisites

- Python 3.8+
- [Google API Client](https://developers.google.com/api-client-library/python)
- [OpenAI API](https://platform.openai.com/docs/)
- [MoviePy](https://zulko.github.io/moviepy/) and [cv2](https://pypi.org/project/opencv-python/) for video processing
- [Pillow](https://python-pillow.org/) for image handling
- [dotenv](https://pypi.org/project/python-dotenv/) for environment management
- A `config.yml` file to manage various configurations

## Installation

1. Clone this repository.
    ```bash
    git clone https://github.com/praveenjune17/pr_agents.git
    cd pr_agents
    ```

2. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

3. Add your **OpenAI API** credentials to the environment using a `.env` file.

4. Configure your `config.yml` file with relevant paths and settings.

## Configuration

- **config.yml**: Update this file with paths, API versions, video privacy settings, scopes, etc.
- **.env file**: Add your OpenAI API key and other sensitive environment variables.

Example `.env` file:

```plaintext
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Run the script to process a video and prepare it for upload:
    ```bash
    python main.py
    ```

2. The script will:
   - Extract frames and audio from the specified video file.
   - Use OpenAI to transcribe audio and generate video titles and descriptions.
   - Create a thumbnail image based on the generated prompt.
   - Upload the video with the generated metadata to YouTube at a specified time.

## Code Overview

- **get_authenticated_service**: Authenticates the YouTube API client using Google OAuth2.
- **upload_video**: Uploads the video to YouTube and sets a custom thumbnail.
- **process_video**: Extracts frames and audio from the video at a specified rate.
- **get_thumbnail**: Uses OpenAI to generate and resize a thumbnail based on a prompt.
- **get_publish_time**: Schedules the publish time based on the specified timezone.

## Example Configuration (config.yml)

```yaml
video_file_folder: "/path/to/video_folder"
video_file: "sample_video.mp4"
thumbnail_file: "thumbnail.jpg"
thumbnail_resolution: [1280, 720]
img_gen_model: "dall-e-3"
video_privacy_status: "public"
secrets_file: "client_secrets.json"
scopes:
  - "https://www.googleapis.com/auth/youtube.upload"
api_version: "v3"
category_id: "22"  # Example category for People & Blogs
time_zone: "Asia/Kolkata"
asr_model: "whisper-1"
llm_prompt: "Generate a YouTube title, description, and thumbnail prompt based on the video content."
date_to_publish:
  publish_yr: 2024
  publish_mth: 12
  publish_day: 15
  publish_hr: 10
  publish_min: 30
```

## License

This project is licensed under the MIT License.

To create the `client_secrets_v2.json` file for accessing the YouTube API, follow these steps:

### Step 1: Set Up a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. If prompted, sign in with your Google account.
3. In the top-left corner, click on the **Select a Project** dropdown and then click **New Project**.
4. Enter a name for your project and click **Create**.

### Step 2: Enable the YouTube Data API

1. Once your project is created, go to the **Dashboard**.
2. In the **APIs & Services** section, click **+ Enable APIs and Services**.
3. Search for **YouTube Data API v3** and select it from the list.
4. Click **Enable** to enable the API for your project.

### Step 3: Set Up OAuth 2.0 Credentials

1. In the **APIs & Services** section, click on **Credentials** in the left-hand menu.
2. Click **Create Credentials** and select **OAuth client ID**.
3. If prompted to set up an OAuth consent screen:
   - Click **Configure Consent Screen**.
   - Select **External** (or **Internal** if using within an organization), and click **Create**.
   - Fill in the necessary fields (e.g., App Name, User Support Email), and click **Save and Continue**.
4. After setting up the consent screen, go back to **Credentials** > **Create Credentials** > **OAuth client ID**.
5. Under **Application type**, select **Desktop app**.
6. Name your OAuth client (e.g., "YouTube Uploader App") and click **Create**.

### Step 4: Download the `client_secrets.json` File

1. After creating the OAuth client, you should see it listed under **OAuth 2.0 Client IDs**.
2. Click the **Download** button next to your client ID to download the `client_secrets.json` file.
3. Save the file to the specified path, renaming it as needed:
   ```
   /{your_folder}/client_secrets_v2.json
   ```

### Step 5: Verify the File Path in Your Code

Ensure your code correctly references this path:

```python
secrets_file = "/{your_folder}/client_secrets_v2.json"
```

The `client_secrets_v2.json` file should now contain the credentials needed to authenticate and access the YouTube Data API.
---

