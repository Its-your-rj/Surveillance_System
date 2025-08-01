import os
import threading
import requests
import imageio_ffmpeg as ffmpeg
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import AzureError, ResourceNotFoundError
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:5000/motion_detected")

# Validate Azure configuration
if not AZURE_STORAGE_CONNECTION_STRING:
    logger.error("Missing AZURE_STORAGE_CONNECTION_STRING environment variable. Please check your .env file.")
    raise ValueError("Azure connection string is required")

if not AZURE_STORAGE_CONTAINER_NAME:
    logger.error("Missing AZURE_STORAGE_CONTAINER_NAME environment variable. Please check your .env file.")
    raise ValueError("Azure container name is required")

# Initialize Azure BlobServiceClient using connection string
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    # Test the connection
    container_properties = container_client.get_container_properties()
    logger.info(f"Successfully connected to Azure Blob Storage container: {AZURE_STORAGE_CONTAINER_NAME}")
    
    # Extract account name and key from connection string for URL generation and SAS
    # The connection string format is: DefaultEndpointsProtocol=https;AccountName=name;AccountKey=key;EndpointSuffix=core.windows.net
    connection_string_parts = dict(part.split('=', 1) for part in AZURE_STORAGE_CONNECTION_STRING.split(';') if '=' in part)
    AZURE_ACCOUNT_NAME = connection_string_parts.get('AccountName')
    AZURE_STORAGE_ACCOUNT_KEY = connection_string_parts.get('AccountKey')
    
    if not AZURE_ACCOUNT_NAME:
        logger.error("Could not extract account name from connection string")
        raise ValueError("Invalid connection string format")
        
    if not AZURE_STORAGE_ACCOUNT_KEY:
        logger.error("Could not extract account key from connection string")
        raise ValueError("Invalid connection string format")
        
except AzureError as e:
    logger.error(f"Failed to connect to Azure Blob Storage: {e}")
    raise

def generate_sas_url(blob_name, expiration_hours=1):
    """
    Generate a SAS (Shared Access Signature) URL for viewing a video blob.
    
    Args:
        blob_name (str): Name of the blob in the container
        expiration_hours (int): Number of hours until the SAS token expires (default: 1)
    
    Returns:
        str: Complete URL with SAS token for viewing the video
    """
    try:
        # Calculate expiration time (1 hour from now)
        expiry = datetime.utcnow() + timedelta(hours=expiration_hours)
        
        # Generate SAS token with read permissions
        sas_token = generate_blob_sas(
            account_name=AZURE_ACCOUNT_NAME,
            container_name=AZURE_STORAGE_CONTAINER_NAME,
            blob_name=blob_name,
            account_key=AZURE_STORAGE_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=expiry
        )
        
        # Construct the complete URL
        sas_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER_NAME}/{blob_name}?{sas_token}"
        
        logger.info(f"Generated SAS URL for {blob_name} with {expiration_hours} hour expiration")
        return sas_url
        
    except Exception as e:
        logger.error(f"Error generating SAS URL for {blob_name}: {e}")
        raise

def upload_to_blob(blob_name, path_to_file):
    """Upload a file to Azure Blob Storage with error handling"""
    try:
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(path_to_file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type='video/mp4'))

        # Clean up local file
        os.remove(path_to_file)

        blob_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER_NAME}/{blob_name}"
        logger.info(f"Successfully uploaded {blob_name} to Azure Blob Storage")
        return blob_url
        
    except FileNotFoundError:
        logger.error(f"File not found: {path_to_file}")
        raise
    except AzureError as e:
        logger.error(f"Azure upload error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise

def handle_detection(path_to_file):
    """Handle motion detection by processing and uploading video"""
    def action_thread(path_to_file):
        try:
            # Check if input file exists before processing
            if not os.path.exists(path_to_file):
                logger.error(f"Input file does not exist: {path_to_file}")
                return

            # Process video with ffmpeg
            output_path = path_to_file.split(".mp4")[0] + "-out.mp4"
            
            # Use imageio_ffmpeg with proper error handling
            try:
                # Get the ffmpeg executable path
                ffmpeg_path = ffmpeg.get_ffmpeg_exe()
                logger.info(f"Using ffmpeg at: {ffmpeg_path}")
                
                # Use subprocess to run ffmpeg command
                import subprocess
                cmd = [
                    ffmpeg_path,
                    '-i', path_to_file,
                    '-vf', 'scale=-1:720',
                    '-y',  # Overwrite output file if it exists
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info(f"Successfully processed video: {path_to_file} -> {output_path}")
                else:
                    logger.error(f"FFmpeg failed with return code {result.returncode}: {result.stderr}")
                    return
                    
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg processing timed out")
                return
            except Exception as e:
                logger.error(f"Error during video processing: {e}")
                return

            # Remove original file after successful processing
            try:
                os.remove(path_to_file)
                logger.info(f"Removed original file: {path_to_file}")
            except OSError as e:
                logger.warning(f"Could not remove original file {path_to_file}: {e}")

            # Upload to Azure
            url = upload_to_blob(os.path.basename(output_path), output_path)
            
            # Send notification
            data = {"url": url}
            try:
                response = requests.post(API_ENDPOINT, json=data, timeout=10)
                response.raise_for_status()
                logger.info("Successfully sent notification")
            except requests.RequestException as e:
                logger.error(f"Failed to send notification: {e}")

        except Exception as e:
            logger.error(f"Error in detection handling: {e}")

    thread = threading.Thread(target=action_thread, args=(path_to_file,))
    thread.start()

def list_videos_in_date_range(start_date, end_date, extension=".mp4"):
    """Fetch videos from Azure Blob Storage within a date range"""
    try:
        # Parse dates and make them timezone-aware (UTC)
        if isinstance(start_date, str):
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_datetime = start_date
            
        if isinstance(end_date, str):
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        else:
            end_datetime = end_date + timedelta(days=1)

        # Make naive datetimes timezone-aware by assuming UTC
        from datetime import timezone
        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
        if end_datetime.tzinfo is None:
            end_datetime = end_datetime.replace(tzinfo=timezone.utc)

        matching_files = []

        # List all blobs in the container
        blobs = container_client.list_blobs()
        
        for blob in blobs:
            try:
                # Check if blob matches criteria
                if blob.name.endswith(extension):
                    # Get creation time from blob properties (already timezone-aware)
                    creation_time = blob.creation_time
                    
                    if start_datetime <= creation_time <= end_datetime:
                        blob_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER_NAME}/{blob.name}"
                        matching_files.append({
                            "url": blob_url, 
                            "date": creation_time.isoformat(),
                            "name": blob.name,
                            "size": blob.size
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing blob {blob.name}: {e}")
                continue

        # Sort by date (newest first)
        matching_files.sort(key=lambda x: x["date"], reverse=True)
        
        logger.info(f"Found {len(matching_files)} videos in date range {start_date} to {end_date}")
        return matching_files
        
    except AzureError as e:
        logger.error(f"Azure error while fetching videos: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while fetching videos: {e}")
        raise

def get_video_metadata(blob_name, include_sas_url=False, sas_expiration_hours=1):
    """Get metadata for a specific video blob"""
    try:
        blob_client = container_client.get_blob_client(blob_name)
        properties = blob_client.get_blob_properties()
        
        metadata = {
            "name": blob_name,
            "size": properties.size,
            "created": properties.creation_time.isoformat(),
            "last_modified": properties.last_modified.isoformat(),
            "content_type": properties.content_settings.content_type,
            "url": f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER_NAME}/{blob_name}"
        }
        
        # Include SAS URL if requested
        if include_sas_url:
            try:
                sas_url = generate_sas_url(blob_name, sas_expiration_hours)
                metadata["sas_url"] = sas_url
                metadata["sas_expiration_hours"] = sas_expiration_hours
            except Exception as e:
                logger.warning(f"Could not generate SAS URL for {blob_name}: {e}")
                metadata["sas_url"] = None
        
        return metadata
    except ResourceNotFoundError:
        logger.error(f"Blob not found: {blob_name}")
        return None
    except AzureError as e:
        logger.error(f"Azure error while getting metadata: {e}")
        return None

def delete_video(blob_name):
    """Delete a video from Azure Blob Storage"""
    try:
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
        logger.info(f"Successfully deleted blob: {blob_name}")
        return True
    except ResourceNotFoundError:
        logger.error(f"Blob not found for deletion: {blob_name}")
        return False
    except AzureError as e:
        logger.error(f"Azure error while deleting blob: {e}")
        return False
