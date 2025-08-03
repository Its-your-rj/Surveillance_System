# Surveillance System with Motion Detection

This repository contains a Python-based surveillance system that uses computer vision and machine learning to detect human motion, record video, upload it to Azure Blob Storage, and send notifications via Twilio. The system is built with a Flask backend to provide a RESTful API for controlling and monitoring the surveillance system. The project was last updated on **August 3, 2025, at 09:35 AM IST**.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Overview](#overview)
   - [Key Components](#key-components)
   - [Dataset](#dataset)
   - [Model Selection](#model-selection)
   - [Preprocessing](#preprocessing)
   - [Inference](#inference)
   - [Post-Processing](#post-processing)
   - [Integration with Surveillance System](#integration-with-surveillance-system)
   - [Chronological Workflow](#chronological-workflow)
   - [Human Identification Mechanism](#human-identification-mechanism)
   - [Error Handling and Robustness](#error-handling-and-robustness)
   - [Why This Pipeline?](#why-this-pipeline)
4. [System Components](#system-components)
   - [Camera Module](#camera-module)
   - [Storage Module](#storage-module)
   - [Notification Module](#notification-module)
   - [Flask Backend](#flask-backend)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [API Endpoints](#api-endpoints)
8. [Environment Variables](#environment-variables)
9. [Dependencies](#dependencies)
10. [Future Improvements](#future-improvements)

## Project Overview
This surveillance system captures video from a specified camera, detects human motion using a pre-trained deep learning model, records video clips when motion is detected, uploads them to Azure Blob Storage, and sends SMS notifications with a link to the video. The system is controlled via a Flask-based REST API, allowing users to arm/disarm the system, retrieve logs, and manage videos. As of August 3, 2025, 09:35 AM IST, the system is fully functional with the described features.

## Architecture
The system follows a modular architecture:
- **Camera Module (`camera.py`)**: Handles video capture, motion detection, and recording.
- **Storage Module (`storage.py`)**: Manages video processing, Azure Blob Storage uploads, and metadata retrieval.
- **Notification Module (`notifications.py`)**: Sends SMS notifications using Twilio.
- **Flask Backend (`main.py`)**: Provides a REST API for controlling the system and retrieving logs.

The system uses threading to handle video processing and uploads asynchronously, ensuring non-blocking operation.

## Machine Learning Pipeline

### Overview
The machine learning (ML) pipeline is designed to detect humans in video frames captured by a camera, using a pre-trained deep learning model. The pipeline is embedded within the `camera.py` module, specifically in the `_process_frame` method, which processes each video frame to detect humans using a pre-trained **MobileNet SSD (Single Shot MultiBox Detector)** model. The pipeline is responsible for identifying human presence in real-time, triggering video recording when a human is detected, and stopping recording when no human is detected for a certain number of frames. It leverages the **COCO dataset** for pre-trained weights and OpenCV's DNN module for inference.

### Key Components
- **Dataset**: COCO dataset (pre-trained model data).
- **Model**: MobileNet SSD for object detection.
- **Preprocessing**: Frame conversion to a blob for model input.
- **Inference**: Running the model to detect objects in the frame.
- **Post-Processing**: Filtering detections to identify humans and drawing bounding boxes.
- **Integration**: Triggering recording and storage based on detection results.

### Dataset
The ML pipeline relies on a pre-trained MobileNet SSD model trained on the **COCO (Common Objects in Context) dataset**.
- **COCO Dataset**:
  - **Description**: COCO is a large-scale dataset containing over 200,000 labeled images with 80 object categories, including the "person" class (class ID 15). It includes diverse scenarios with varying lighting, backgrounds, and human poses.
  - **Why COCO?**:
    - **Diversity**: The dataset covers a wide range of environments, making the model robust for real-world surveillance applications.
    - **Annotations**: COCO provides bounding box annotations for objects, which are critical for training object detection models like MobileNet SSD.
    - **Standardization**: The COCO dataset is widely used, ensuring compatibility with pre-trained models and tools like OpenCV's DNN module.
  - **Relevance to Human Detection**: The "person" class is explicitly labeled in COCO, allowing the model to accurately identify humans in video frames.
  - **Pre-trained Model**: The model uses weights (`mobilenet_iter_73000.caffemodel`) and configuration (`config.txt`) trained on COCO, loaded in the `Camera.__init__` method:
    ```python
    self.net = cv.dnn.readNetFromCaffe('models/config.txt', 'models/mobilenet_iter_73000.caffemodel')
    ```

### Model Selection
The pipeline uses **MobileNet SSD**, a lightweight and efficient deep learning model designed for real-time object detection.
- **Why MobileNet SSD?**:
  - **Efficiency**: MobileNet is optimized for low-latency inference, making it suitable for real-time video processing on resource-constrained devices.
  - **Accuracy**: SSD provides accurate bounding box predictions and class probabilities in a single forward pass, balancing speed and precision.
  - **Lightweight Architecture**: MobileNet uses depth-wise separable convolutions, reducing computational complexity compared to heavier models like YOLO or Faster R-CNN.
  - **Pre-trained Weights**: The model is pre-trained on COCO, eliminating the need for custom training and enabling immediate use for human detection.
- **Model Loading**:
  - In the `Camera.__init__` method, the model is loaded using OpenCV's DNN module:
    ```python
    self.net = cv.dnn.readNetFromCaffe('models/config.txt', 'models/mobilenet_iter_73000.caffemodel')
    ```
  - The `config.txt` file defines the network architecture, while `mobilenet_iter_73000.caffemodel` contains the trained weights after 73,000 iterations on COCO.

### Preprocessing
Before a frame can be processed by the MobileNet SSD model, it must be preprocessed into a format suitable for the neural network.
- **Process**:
  - The `_process_frame` method converts each video frame into a **blob**, a standardized 4D tensor (batch size, channels, height, width) expected by the model.
  - Code:
    ```python
    blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    ```
  - **Parameters**:
    - `frame`: The input video frame (a 3D NumPy array with shape [height, width, 3] for RGB).
    - `0.007843`: Scaling factor (1/127.5) to normalize pixel values.
    - `(300, 300)`: Resizes the frame to 300x300 pixels, the input size expected by MobileNet SSD.
    - `127.5`: Mean subtraction value applied to each channel to center the data (standard preprocessing for MobileNet SSD).
  - **Output**: The blob is a 4D tensor with shape [1, 3, 300, 300], representing a single image with 3 color channels.
- **Why Preprocessing?**:
  - **Standardization**: The model was trained on images with specific preprocessing (resized to 300x300, normalized, and mean-subtracted), so the input frame must match these conditions.
  - **Efficiency**: Resizing to 300x300 reduces computational load while maintaining sufficient detail for detection.

### Inference
The preprocessed blob is fed into the MobileNet SSD model to detect objects in the frame.
- **Process**:
  - The blob is set as the input to the neural network:
    ```python
    self.net.setInput(blob)
    ```
  - The model performs a forward pass to generate detection outputs:
    ```python
    detections = self.net.forward()
    ```
  - **Output Format**:
    - The `detections` output is a 4D tensor with shape [1, 1, N, 7], where:
      - `N`: Number of detected objects.
      - Each detection is a 7-element vector: `[batch_id, class_id, confidence, x_min, y_min, x_max, y_max]`.
      - `class_id`: Index of the detected object class (e.g., 15 for "person" in COCO).
      - `confidence`: Probability score indicating the likelihood of the detection.
      - `x_min, y_min, x_max, y_max`: Normalized bounding box coordinates (0 to 1).
- **Why Forward Pass?**:
  - The forward pass computes feature maps using the MobileNet backbone and applies SSD's detection heads to predict bounding boxes and class probabilities.
  - The single-shot nature of SSD allows simultaneous detection of multiple objects in one pass, ensuring real-time performance.

### Post-Processing
The raw detections are processed to identify humans and draw bounding boxes (optional for debugging).
- **Process**:
  - The `_process_frame` method iterates through the detections:
    ```python
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        if idx == 15 and confidence > 0.5:  # 15 is the "person" class in COCO
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            person_detected = True
    ```
  - **Steps**:
    1. **Extract Confidence**: Retrieve the confidence score for each detection.
    2. **Filter by Class**: Check if the class ID is 15 (person in COCO).
    3. **Confidence Threshold**: Only consider detections with confidence > 0.5 to reduce false positives.
    4. **Bounding Box Scaling**: Scale the normalized coordinates (`x_min, y_min, x_max, y_max`) to the original frame dimensions (width, height).
    5. **Draw Bounding Box**: Draw a green rectangle around the detected human (optional, can be commented out for production).
    6. **Flag Detection**: Set `person_detected = True` if at least one human is detected.
- **Output**:
  - `person_detected`: Boolean indicating whether a human was detected.
  - `processed_frame`: The frame with bounding boxes drawn (if enabled).
- **Why Post-Processing?**:
  - **Filtering**: Ensures only high-confidence human detections are considered.
  - **Visualization**: Bounding boxes help with debugging and verification (though disabled in production to save resources).
  - **State Management**: The `person_detected` flag drives the recording logic.

### Integration with Surveillance System
The ML pipeline is tightly integrated with the surveillance system's recording and storage mechanisms.
- **Recording Logic** (in `Camera.run`):
  - The `run` method runs a continuous loop while the system is armed (`self.armed`):
    ```python
    while self.armed:
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            continue
        person_detected, processed_frame = self._process_frame(frame)
        if person_detected:
            non_detected_counter = 0
            if self.out is None:
                self._start_recording(frame)
            if self.out is not None:
                self.out.write(frame)
        else:
            non_detected_counter += 1
            if non_detected_counter >= 50 and self.out is not None:
                self._stop_recording()
    ```
  - **Workflow**:
    1. **Frame Capture**: Read a frame from the camera using `self.cap.read()`.
    2. **Detection**: Call `_process_frame` to check for humans.
    3. **Start Recording**: If a human is detected and no recording is active (`self.out is None`), call `_start_recording` to initialize a new video file:
       ```python
       now = datetime.datetime.now()
       formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
       self.current_recording_name = f'{formatted_now}.mp4'
       fourcc = cv.VideoWriter_fourcc(*'mp4v')
       self.out = cv.VideoWriter(self.current_recording_name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
       ```
    4. **Write Frame**: Write the frame to the video file if recording is active.
    5. **Stop Recording**: If no human is detected for 50 frames (`non_detected_counter >= 50`), call `_stop_recording`:
       ```python
       if self.out is not None:
           self.out.release()
           self.out = None
           if self.current_recording_name:
               handle_detection(self.current_recording_name)
               self.current_recording_name = None
       ```
    6. **Storage Integration**: The `handle_detection` function (in `storage.py`) processes the video (scales to 720p using FFmpeg), uploads it to Azure Blob Storage, and sends a notification.
- **Storage and Notification**:
  - **Video Processing**: The `handle_detection` function scales the video to 720p using FFmpeg to ensure consistency and reduce storage size:
    ```python
    cmd = [ffmpeg_path, '-i', path_to_file, '-vf', 'scale=-1:720', '-y', output_path]
    ```
  - **Upload**: The processed video is uploaded to Azure Blob Storage with a `video/mp4` content type:
    ```python
    blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type='video/mp4'))
    ```
  - **Notification**: A notification is sent via Twilio with a URL to the uploaded video:
    ```python
    response = requests.post(API_ENDPOINT, json={"url": url}, timeout=10)
    ```
- **Why Integration?**:
  - **Automation**: The pipeline automatically triggers recording and storage based on ML detections, ensuring only relevant footage is saved.
  - **Scalability**: Asynchronous processing (via threading in `handle_detection`) prevents the ML pipeline from blocking the main surveillance loop.
  - **User Interaction**: The Flask API allows users to control the system and access videos, leveraging the ML pipeline's outputs.

### Chronological Workflow
1. **Initialization** (`Camera.__init__`):
   - Load the MobileNet SSD model with COCO-trained weights.
   - Initialize the camera (`cv.VideoCapture(camera_index)`).
2. **Arming the System** (`Camera.arm`):
   - Start a daemon thread to run the surveillance loop (`run`).
3. **Frame Capture** (`Camera.run`):
   - Continuously read frames from the camera.
4. **Preprocessing** (`_process_frame`):
   - Convert each frame to a 300x300 blob.
5. **Inference** (`_process_frame`):
   - Run the MobileNet SSD model to detect objects.
6. **Post-Processing** (`_process_frame`):
   - Filter for human detections (class ID 15, confidence > 0.5).
   - Draw bounding boxes (optional) and flag human presence.
7. **Recording** (`Camera.run`):
   - Start recording if a human is detected and no recording is active.
   - Write frames to the video file.
   - Stop recording after 50 frames without human detection.
8. **Storage and Notification** (`_stop_recording` → `handle_detection`):
   - Process the video with FFmpeg.
   - Upload to Azure Blob Storage.
   - Send an SMS notification with the video URL.

### Human Identification Mechanism
The system identifies humans through the following steps:
- **Class ID Check**: The MobileNet SSD model outputs a class ID for each detection. The system filters for `idx == 15`, which corresponds to the "person" class in the COCO dataset.
- **Confidence Threshold**: A detection is only considered valid if its confidence score exceeds 0.5, reducing false positives (e.g., mistaking objects for humans).
- **Bounding Box**: The model provides normalized bounding box coordinates, which are scaled to the frame's dimensions to locate the human in the image.
- **Robustness**:
  - The COCO-trained model is robust to variations in human appearance, pose, and lighting due to the dataset's diversity.
  - The 0.5 confidence threshold balances sensitivity and specificity, ensuring reliable detection without excessive false positives.

### Error Handling and Robustness
- **Logging**: The pipeline uses Python's `logging` module to log errors (e.g., model loading failures, frame processing issues) for debugging.
- **Exception Handling**: Each step (model loading, frame capture, preprocessing, inference, recording) is wrapped in try-except blocks to prevent crashes.
  ```python
  try:
      self.net = cv.dnn.readNetFromCaffe('models/config.txt', 'models/mobilenet_iter_73000.caffemodel')
  except Exception as e:
      logger.error(f"Failed to load neural network model: {e}")
      raise
  ```
- **Camera Reinitialization**: If frame capture fails, the system logs a warning and continues the loop, ensuring resilience.
- **Threading**: The surveillance loop runs in a daemon thread, allowing clean program termination.

### Why This Pipeline?
- **Real-Time Performance**: MobileNet SSD's lightweight architecture ensures low-latency detection, critical for surveillance.
- **Pre-trained Model**: Using a COCO-trained model eliminates the need for custom training, saving time and resources.
- **Integration**: The pipeline seamlessly integrates with recording, storage, and notification systems, creating a cohesive surveillance solution.
- **Scalability**: Threading and asynchronous processing ensure the system can handle multiple tasks without performance degradation.

## System Components

### Camera Module (`camera.py`)
- **Purpose**: Captures video, performs human detection, and manages video recording.
- **Key Functions**:
  - `__init__`: Initializes the camera and loads the MobileNet SSD model.
  - `arm`/`disarm`: Starts/stops the surveillance loop.
  - `_process_frame`: Processes each frame for human detection using the MobileNet SSD model.
  - `_start_recording`/`_stop_recording`: Manages video recording and triggers uploads via the `storage` module.
  - `run`: Main surveillance loop that captures and processes frames.
- **Threading**: Uses a daemon thread to run the surveillance loop, ensuring the main program can exit cleanly.
- **Error Handling**: Logs errors for camera initialization, model loading, and frame processing.

### Storage Module (`storage.py`)
- **Purpose**: Processes recorded videos, uploads them to Azure Blob Storage, and manages video metadata.
- **Key Functions**:
  - `upload_to_blob`: Uploads videos to Azure Blob Storage with the correct content type (`video/mp4`).
  - `handle_detection`: Processes videos using FFmpeg (scales to 720p height), uploads them, and sends notifications.
  - `generate_sas_url`: Generates a Shared Access Signature (SAS) URL for secure video access.
  - `list_videos_in_date_range`: Retrieves video metadata within a specified date range.
  - `get_video_metadata`/`delete_video`: Manages video metadata and deletion.
- **Threading**: Uses threading to process and upload videos asynchronously.
- **FFmpeg**: Scales videos to 720p for consistency and storage efficiency.

### Notification Module (`notifications.py`)
- **Purpose**: Sends SMS notifications via Twilio when motion is detected.
- **Key Functions**:
  - `send_notification`: Sends an SMS with a timestamp and video URL.
  - `test_notification`: Tests the notification system with a dummy URL.
- **Twilio Integration**: Uses environment variables for configuration (Account SID, Auth Token, etc.).
- **Error Handling**: Logs errors for Twilio initialization and message sending.

### Flask Backend (`main.py`)
- **Purpose**: Provides a REST API to control the surveillance system and retrieve logs.
- **Key Endpoints**:
  - `/arm`: Arms the surveillance system.
  - `/disarm`: Disarms the system.
  - `/get-armed`: Retrieves the armed status.
  - `/motion_detected`: Handles motion detection notifications.
  - `/get-logs`: Retrieves video logs for a date range.
  - `/get-recent-logs`: Retrieves recent logs (default: last 7 days).
  - `/video/<blob_name>/metadata`: Retrieves metadata for a specific video.
  - `/video/<blob_name>` (DELETE): Deletes a video.
  - `/health`: Performs a health check.
- **CORS**: Enabled to allow cross-origin requests.
- **Error Handling**: Includes handlers for 404 and 500 errors.

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory with the following:
   ```env
   CAMERA_INDEX=1
   AZURE_STORAGE_CONNECTION_STRING=<your-azure-connection-string>
   AZURE_STORAGE_CONTAINER_NAME=<your-container-name>
   TWILIO_ACCOUNT_SID=<your-twilio-account-sid>
   TWILIO_AUTH_TOKEN=<your-twilio-auth-token>
   TWILIO_SEND_NUMBER=<your-twilio-sender-number>
   TWILIO_RECEIVE_NUMBER=<your-twilio-receiver-number>
   API_ENDPOINT=http://localhost:5000/motion_detected
   ```
4. **Install FFmpeg**:
   - Install FFmpeg on your system (required for video processing).
   - Ensure `imageio_ffmpeg` can locate the FFmpeg executable.
5. **Download Model Files**:
   - Place `mobilenet_iter_73000.caffemodel` and `config.txt` in the `models/` directory.
   - These files are available from the OpenCV repository or pre-trained model sources.
6. **Run the Application**:
   ```bash
   python main.py
   ```
   The Flask server will start on `http://0.0.0.0:5000`.

## Usage
1. **Start the Server**:
   Run `python main.py` to start the Flask server.
2. **Control the System**:
   - Arm the system: `curl -X POST http://localhost:5000/arm`
   - Disarm the system: `curl -X POST http://localhost:5000/disarm`
   - Check armed status: `curl http://localhost:5000/get-armed`
3. **Retrieve Logs**:
   - Get logs for a date range: `curl "http://localhost:5000/get-logs?startDate=2023-01-01&endDate=2023-01-31"`
   - Get recent logs: `curl http://localhost:5000/get-recent-logs?days=7`
4. **Manage Videos**:
   - Get video metadata: `curl http://localhost:5000/video/<blob_name>/metadata`
   - Delete a video: `curl -X DELETE http://localhost:5000/video/<blob_name>`
5. **Health Check**:
   - Check system health: `curl http://localhost:5000/health`

## API Endpoints
| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/arm` | POST | Arms the surveillance system | None |
| `/disarm` | POST | Disarms the system | None |
| `/get-armed` | GET | Retrieves armed status | None |
| `/motion_detected` | POST | Handles motion detection notifications | JSON: `{ "url": "<video-url>" }` |
| `/get-logs` | GET | Retrieves video logs for a date range | Query: `startDate`, `endDate` (YYYY-MM-DD) |
| `/get-recent-logs` | GET | Retrieves recent logs | Query: `days` (default: 7) |
| `/video/<blob_name>/metadata` | GET | Retrieves video metadata | Path: `blob_name` |
| `/video/<blob_name>` | DELETE | Deletes a video | Path: `blob_name` |
| `/health` | GET | Performs a health check | None |

## Environment Variables
| Variable | Description |
|----------|-------------|
| `CAMERA_INDEX` | Camera index for OpenCV (default: 1) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string |
| `AZURE_STORAGE_CONTAINER_NAME` | Azure container name |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_SEND_NUMBER` | Twilio sender phone number |
| `TWILIO_RECEIVE_NUMBER` | Twilio receiver phone number |
| `API_ENDPOINT` | API endpoint for notifications (default: `http://localhost:5000/motion_detected`) |

## Dependencies
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Flask
- Flask-CORS
- azure-storage-blob
- twilio
- python-dotenv
- imageio-ffmpeg
- requests
- FFmpeg (system installation)

Install dependencies using:
```bash
pip install opencv-python numpy flask flask-cors azure-storage-blob twilio python-dotenv imageio-ffmpeg requests
```

## Future Improvements
- **Real-Time Streaming**: Add support for real-time video streaming via WebRTC.
- **Advanced Detection**: Incorporate multi-class detection or activity recognition.
- **Improved Notifications**: Add support for email or push notifications.
- **User Interface**: Develop a frontend dashboard for easier system control and video playback.
- **Model Optimization**: Fine-tune the MobileNet SSD model for specific surveillance scenarios.