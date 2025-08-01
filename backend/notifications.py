from twilio.rest import Client
from twilio.base.exceptions import TwilioException
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Twilio Configuration
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
SENDER = os.getenv("TWILIO_SEND_NUMBER")
RECEIVER = os.getenv("TWILIO_RECEIVE_NUMBER")

# Validate Twilio configuration
if not all([ACCOUNT_SID, AUTH_TOKEN, SENDER, RECEIVER]):
    logger.warning("Twilio configuration incomplete. Notifications will be disabled.")
    client = None
else:
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        logger.info("Twilio client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Twilio client: {e}")
        client = None

def send_notification(url):
    """Send SMS notification via Twilio"""
    if client is None:
        logger.warning("Twilio client not available. Skipping notification.")
        return False
    
    try:
        now = datetime.now()
        formatted_now = now.strftime("%d/%m/%y %H:%M:%S")
        
        message_body = f"Person motion detected @{formatted_now}: {url}"
        
        message = client.messages.create(
            body=message_body,
            from_=SENDER,
            to=RECEIVER
        )
        
        logger.info(f"Notification sent successfully. SID: {message.sid}")
        return True
        
    except TwilioException as e:
        logger.error(f"Twilio error sending notification: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending notification: {e}")
        return False

def test_notification():
    """Test the notification system"""
    if client is None:
        logger.error("Twilio client not available for testing")
        return False
    
    try:
        test_url = "https://example.com/test-video.mp4"
        success = send_notification(test_url)
        
        if success:
            logger.info("Notification test successful")
        else:
            logger.error("Notification test failed")
            
        return success
        
    except Exception as e:
        logger.error(f"Error during notification test: {e}")
        return False