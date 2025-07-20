import os
import threading
import requests
import ffmpeg
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Configuration
AZURE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
API_ENDPOINT = os.getenv("API_ENDPOINT")

# Initialize Azure BlobServiceClient
AZURE_CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

def upload_to_blob(blob_name, path_to_file):
    blob_client = container_client.get_blob_client(blob_name)
    
    with open(path_to_file, "rb") as data:
        blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type='video/mp4'))

    os.remove(path_to_file)

    blob_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_name}"
    print(f"A new file named {blob_name} was uploaded to container {AZURE_CONTAINER_NAME}")
    return blob_url

def handle_detection(path_to_file):
    def action_thread(path_to_file):
        output_path = path_to_file.split(".mp4")[0] + "-out.mp4"
        ffmpeg.input(path_to_file).output(output_path, vf='scale=-1:720').run()
        os.remove(path_to_file)

        url = upload_to_blob(os.path.basename(output_path), output_path)
        data = {"url": url}
        requests.post(API_ENDPOINT, json=data)

    thread = threading.Thread(target=action_thread, args=(path_to_file,))
    thread.start()

def list_videos_in_date_range(start_date, end_date, extension=".mp4"):
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

    matching_files = []

    blobs = container_client.list_blobs()
    for blob in blobs:
        if blob.name.endswith(extension) and start_datetime <= blob.creation_time <= end_datetime:
            blob_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob.name}"
            matching_files.append({"url": blob_url, "date": blob.creation_time})

    return matching_files
