from camera import Camera
from notifications import send_notification
from storage import list_videos_in_date_range, get_video_metadata, delete_video
from flask_cors import CORS
from flask import Flask, jsonify, request
import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Get camera index from environment variable or default to 0
camera_index = int(os.getenv('CAMERA_INDEX', 1))
camera = Camera(camera_index=camera_index)

@app.route('/arm', methods=['POST'])
def arm():
    """Arm the surveillance system"""
    try:
        camera.arm()
        logger.info("Surveillance system armed")
        return jsonify(message="System armed."), 200
    except Exception as e:
        logger.error(f"Error arming system: {e}")
        return jsonify(error="Failed to arm system"), 500

@app.route('/disarm', methods=['POST'])
def disarm():
    """Disarm the surveillance system"""
    try:
        camera.disarm()
        logger.info("Surveillance system disarmed")
        return jsonify(message="System disarmed."), 200
    except Exception as e:
        logger.error(f"Error disarming system: {e}")
        return jsonify(error="Failed to disarm system"), 500

@app.route('/get-armed', methods=['GET'])
def get_armed():
    """Get the current armed status"""
    try:
        return jsonify(armed=camera.armed), 200
    except Exception as e:
        logger.error(f"Error getting armed status: {e}")
        return jsonify(error="Failed to get armed status"), 500

@app.route('/motion_detected', methods=['POST'])
def motion_detected():
    """Handle motion detection notifications"""
    try:
        data = request.get_json()

        if 'url' in data:
            logger.info(f"Motion detected, URL: {data['url']}")
            send_notification(data["url"])
            return jsonify(message="Notification sent"), 201
        else:
            logger.warning("'url' not in incoming data")
            return jsonify(error="Missing URL in request"), 400

    except Exception as e:
        logger.error(f"Error handling motion detection: {e}")
        return jsonify(error="Failed to handle motion detection"), 500

@app.route("/get-logs")
def get_logs():
    """Get video logs from Azure Blob Storage within a date range"""
    try:
        start_date = request.args.get("startDate")  # y-m-d
        end_date = request.args.get("endDate")  # y-m-d

        # Validate date parameters
        if not start_date or not end_date:
            return jsonify(error="startDate and endDate parameters are required"), 400

        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify(error="Invalid date format. Use YYYY-MM-DD"), 400

        logs = list_videos_in_date_range(start_date, end_date)
        return jsonify({"logs": logs}), 200

    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify(error="Failed to fetch logs"), 500

@app.route("/get-recent-logs")
def get_recent_logs():
    """Get recent video logs (last 7 days by default)"""
    try:
        days = request.args.get("days", "7")
        
        try:
            days = int(days)
        except ValueError:
            return jsonify(error="Invalid days parameter"), 400

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logs = list_videos_in_date_range(start_date, end_date)
        return jsonify({"logs": logs}), 200

    except Exception as e:
        logger.error(f"Error fetching recent logs: {e}")
        return jsonify(error="Failed to fetch recent logs"), 500

@app.route("/video/<blob_name>/metadata")
def get_video_metadata_endpoint(blob_name):
    """Get metadata for a specific video"""
    try:
        metadata = get_video_metadata(blob_name)
        if metadata:
            return jsonify(metadata), 200
        else:
            return jsonify(error="Video not found"), 404

    except Exception as e:
        logger.error(f"Error fetching video metadata: {e}")
        return jsonify(error="Failed to fetch video metadata"), 500

@app.route("/video/<blob_name>", methods=['DELETE'])
def delete_video_endpoint(blob_name):
    """Delete a specific video from Azure Blob Storage"""
    try:
        success = delete_video(blob_name)
        if success:
            return jsonify(message="Video deleted successfully"), 200
        else:
            return jsonify(error="Video not found or could not be deleted"), 404

    except Exception as e:
        logger.error(f"Error deleting video: {e}")
        return jsonify(error="Failed to delete video"), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "armed": camera.armed
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify(error="Endpoint not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify(error="Internal server error"), 500

if __name__ == "__main__":
    logger.info("Starting surveillance system backend...")
    app.run(host='0.0.0.0', port=5000, debug=False)

