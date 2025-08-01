import cv2 as cv
import numpy as np
import threading
import datetime
import logging
from storage import handle_detection

# Configure logging
logger = logging.getLogger(__name__)

class Camera:

    def __init__(self, camera_index):
        self.armed = False
        self.camera_thread = None
        self.cap = None
        self.out = None
        self.current_recording_name = None
        self.camera_index = camera_index
        self.cap = cv.VideoCapture(self.camera_index)
        # Load the neural network model
        try:
            self.net = cv.dnn.readNetFromCaffe('models/config.txt', 'models/mobilenet_iter_73000.caffemodel')
            logger.info("Neural network model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load neural network model: {e}")
            raise
    
    def arm(self):
        """Arm the surveillance system"""
        if not self.armed and not self.camera_thread:
            self.camera_thread = threading.Thread(target=self.run)
            self.camera_thread.daemon = True  # Make thread daemon so it stops when main program exits

        self.camera_thread.start()
        if not self.armed:
            self.armed = True
            logger.info("Camera armed and surveillance started")

    def disarm(self):
        """Disarm the surveillance system"""
        self.armed = False
        if self.camera_thread:
            self.camera_thread = None
        logger.info("Camera disarmed")

    def _initialize_camera(self):
        """Initialize the camera capture"""
        try:
            # Use the specified camera index
            self.cap = cv.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                logger.info(f"Camera initialized successfully on index {self.camera_index}")
                return True
            else:
                logger.error(f"Failed to open camera on index {self.camera_index}")
                return False
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False

    def _process_frame(self, frame):
        """Process a single frame for person detection"""
        try:
            # Create blob from image
            blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            person_detected = False
            
            for i in range(detections.shape[2]):
                # Extract the confidence
                confidence = detections[0, 0, i, 2]
                
                # Get the label for the class number
                idx = int(detections[0, 0, i, 1])
                
                # Check if the detection is of a person and its confidence is above threshold
                if idx == 15 and confidence > 0.5:  # 15 is the class index for person in COCO dataset
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    person_detected = True
                    
            return person_detected, frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return False, frame

    def _start_recording(self, frame):
        """Start recording a new video"""
        try:
            now = datetime.datetime.now()
            formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
            logger.info(f"Person motion detected at {formatted_now}")
            
            self.current_recording_name = f'{formatted_now}.mp4'
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            
            self.out = cv.VideoWriter(
                self.current_recording_name, 
                fourcc, 
                20.0, 
                (frame.shape[1], frame.shape[0])
            )
            
            logger.info(f"Started recording: {self.current_recording_name}")
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")

    def _stop_recording(self):
        """Stop the current recording and process it"""
        try:
            if self.out is not None:
                self.out.release()
                self.out = None
                
                if self.current_recording_name:
                    logger.info(f"Stopped recording: {self.current_recording_name}")
                    handle_detection(self.current_recording_name)
                    self.current_recording_name = None
                    
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")

    def run(self):
        """Main surveillance loop"""
        person_detected = False
        non_detected_counter = 0
        
        # Initialize camera
        if not self._initialize_camera():
            logger.error("Failed to initialize camera")
            self.armed = False
            return

        logger.info("Surveillance loop started")
        Camera.cap = cv.VideoCapture(1)
        try:
            while self.armed:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                # Process frame for person detection
                person_detected, processed_frame = self._process_frame(frame)
                
                # Handle recording based on detection
                if person_detected:
                    non_detected_counter = 0  # Reset counter
                    
                    # Start recording if not already recording
                    if self.out is None:
                        self._start_recording(frame)
                    
                    # Write frame to video
                    if self.out is not None:
                        self.out.write(frame)
                        
                else:
                    non_detected_counter += 1
                    
                    # Stop recording after 50 frames without detection
                    if non_detected_counter >= 50 and self.out is not None:
                        self._stop_recording()
                
                # Optional: Display the frame (for debugging)
                # cv.imshow('Surveillance', processed_frame)
                # if cv.waitKey(1) & 0xFF == ord('q'):
                #     break
                    
        except Exception as e:
            logger.error(f"Error in surveillance loop: {e}")
            
        finally:
            # Cleanup
            self._stop_recording()
            if self.cap is not None:
                self.cap.release()
            logger.info("Surveillance loop ended")

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.cap is not None:
                self.cap.release()
            if self.out is not None:
                self.out.release()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

