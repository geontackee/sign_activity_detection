import cv2 #OpenCV for computer vision tasks
import numpy as np
import argparse #For parsing command line arguments
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time #For fps tracking
from collections import deque #Data structure for sliding window
from mediapipe.framework.formats import landmark_pb2

model_path = '/Users/chatsign/Desktop/sad/hand_landmarker.task'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path=model_path)

class RealtimeSigningDetector:
    def __init__(self, velocity_threshold, moving_average_window):
        # Initialize parameters
        self.velocity_threshold = velocity_threshold
        self.moving_average_window = moving_average_window
        
        # Initialize mediapipe model
        options = vision.HandLandmarkerOptions(
            running_mode = VisionRunningMode.VIDEO,
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence =0.5
        )

        self.hands = vision.HandLandmarker.create_from_options(options)

        # Velocity tracking (left and right hand)
        self.left_velocity_history = deque(maxlen=self.moving_average_window)
        self.right_velocity_history = deque(maxlen=self.moving_average_window)
        self.left_prev_keypoints = None
        self.right_prev_keypoints = None
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.thickness = 2

    
    def hand_velocity(self, curr_kpts, prev_kpts): #use also z value (depth)
        """Calculate velocity of the "important" keypoints """
        """ 0 : WRIST
            4 : THUMB_TIP
            8 : INDEX_FINGER_TIP
            16 : RING_FINGER_TIP
            20 : PINKY_TIP"""
        important_index = [0, 4, 8, 16, 20]

        cur_landmarks_array = np.array([[curr_kpts[i].x, curr_kpts[i].y, curr_kpts[i].z] for i in important_index])
        if prev_kpts is None:
            # No previous, so velocity is distance from origin (0,0)
            velocities = np.linalg.norm(cur_landmarks_array, axis=1)
        else:
            prev_landmarks_array = np.array([[prev_kpts[i].x, prev_kpts[i].y, prev_kpts[i].z] for i in important_index])
            velocities = np.linalg.norm(cur_landmarks_array - prev_landmarks_array, axis=1)
        
        # Return the mean velocity across all important landmarks
        return np.mean(velocities)
    
    def get_moving_average_velocity(self, handedness):
        """Get moving average velocity for a person"""
        if handedness == 'Left':
            return np.mean(self.left_velocity_history)
        else:
            return np.mean(self.right_velocity_history)
    
    def draw_keypoints(self, frame, keypoints, scores=None):
        """Draw face, left hand, and right hand keypoints with size scaled to resolution"""
        height, width = frame.shape[:2]
        
        # 🔧 Set circle radius based on resolution
        radius = max(2, int(height * 0.005))  # 0.5% of height, minimum radius = 2

        def draw_points(indices, color):
            for idx in indices:
                if np.isnan(keypoints[idx]).any():
                    continue
                if scores is not None and scores[idx] < VALID_THRESHOLD:
                    continue
                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                cv2.circle(frame, (x, y), radius, color, -1)

        # 🎯 Draw keypoints
        draw_points(FACE, (255, 255, 0))       # Yellow for face
        draw_points(LEFT_HAND, (0, 255, 255))  # Cyan for left hand
        draw_points(RIGHT_HAND, (255, 0, 255)) # Magenta for right hand
    
    def process_frame(self, frame, timestamp_ms):
        """Process a single frame and detect signing""" 

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #converting to accepted format for mediapipe
        results = self.hands.detect_for_video(mp_image, timestamp_ms) #process frame
        detected_left = False
        detected_right = False

        for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            label = handedness[0].category_name
            if label == 'Left':
                self.left_velocity_history.append(self.hand_velocity(hand_landmarks, self.left_prev_keypoints))
                self.left_prev_keypoints = hand_landmarks
                detected_left = True
            else:
                self.right_velocity_history.append(self.hand_velocity(hand_landmarks, self.right_prev_keypoints))
                self.right_prev_keypoints = hand_landmarks
                detected_right = True
        
        if not detected_left:
            self.left_velocity_history.append(0)
            self.left_prev_keypoints = None
        if not detected_right:
            self.right_velocity_history.append(0)
            self.right_prev_keypoints = None
        
        # moving average velocity for left and right hand
        left_avg_velocity = self.get_moving_average_velocity('Left')
        right_avg_velocity = self.get_moving_average_velocity('Right')
        avg_velocity = 0

        if left_avg_velocity == 0 and right_avg_velocity != 0:
            avg_velocity = right_avg_velocity
        elif right_avg_velocity == 0 and left_avg_velocity != 0:
            avg_velocity = left_avg_velocity
        else:
            avg_velocity = (left_avg_velocity + right_avg_velocity) / 2

        is_signing = False
        if avg_velocity > self.velocity_threshold:
            is_signing = True
        
        signing_status  = {}
        signing_status['is_signing'] = is_signing
        signing_status['avg_velocity'] = avg_velocity
            
        return frame, signing_status, results

    def draw_landmarks_on_image(self, image, detection_result):
        for hand_landmarks in detection_result.hand_landmarks:
            landmark_list = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(
                        x=lm.x, y=lm.y, z=lm.z
                    ) for lm in hand_landmarks
                ]
            )
            mp_drawing.draw_landmarks(
                image,
                landmark_list,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
        return image
    
    def draw_results(self, frame, signing_status):
        """Draw signing detection results on frame with resolution-aware scaling"""
        height, width = frame.shape[:2]

        # 🔧 Scale text sizes and spacing based on height
        base_font_scale = height / 720.0  # Normalize based on 720p
        font_scale_header = max(0.8, base_font_scale * 1.2)
        font_scale_small = max(0.5, base_font_scale * 0.7)
        font_scale_normal = max(0.6, base_font_scale * 0.9)
        line_spacing = int(height * 0.035)  # Space between lines
        margin = int(height * 0.02)
        self.thickness = max(1, int(height / 500))

        # Header
        cv2.putText(frame, "Real-time Signing Detection", (margin, margin + line_spacing),
                    self.font, font_scale_header, (255, 255, 255), self.thickness)

        # Settings info
        cv2.putText(frame, f"Threshold: {self.velocity_threshold} pixels/frame",
                    (margin, margin + 2 * line_spacing),
                    self.font, font_scale_small, (255, 255, 255), self.thickness)

        cv2.putText(frame, f"Moving Average Window: {self.moving_average_window} frames",
                    (margin, margin + 3 * line_spacing),
                    self.font, font_scale_small, (255, 255, 255), self.thickness)

        # Person-by-person info
        y_offset = margin + 5 * line_spacing
        is_signing = signing_status['is_signing']
        avg_velocity = signing_status['avg_velocity']
        color = (0, 255, 0) if is_signing else (0, 0, 255)
        status_text = "SIGNING" if is_signing else "NOT SIGNING"

        # Status
        cv2.putText(frame, f"  Person {status_text}",
                    (margin, y_offset), self.font, font_scale_normal, color, self.thickness)

        # Velocity info
        cv2.putText(frame, f"  Avg Velocity: {avg_velocity} pixels/frame",
                    (margin, y_offset + line_spacing),
                    self.font, font_scale_small, (255, 255, 255), 1)

        y_offset += 3 * line_spacing

        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset",
                    (margin, height - margin),
                    self.font, font_scale_small, (255, 255, 255), 1)

        return frame

    
    def run(self, camera_id=0):
        """Run the real-time signing detection"""
        # Try different camera backends
        backends = [
            cv2.CAP_ANY,  # Auto-detect
            cv2.CAP_AVFOUNDATION,  # macOS specific
            cv2.CAP_V4L2,  # Linux
            cv2.CAP_DSHOW  # Windows
        ]
        
        cap = None
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"✓ Camera opened successfully with backend {backend}")
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception as e:
                print(f"Backend {backend} failed: {e}")
                if cap:
                    cap.release()
                    cap = None
        
        if cap is None or not cap.isOpened():
            print(f"Error: Could not open camera {camera_id} with any backend")
            print("Please check camera permissions in System Preferences > Security & Privacy > Camera")
            return
        
        print("Starting real-time signing detection...")
        print(f"Velocity threshold: {self.velocity_threshold} pixels/frame")
        print(f"Moving average window: {self.moving_average_window} frames")
        print("Press 'q' to quit, 'r' to reset")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            timestamp_ms = int(frame_count * (1000 / 15))
            processed_frame, signing_status, results = self.process_frame(frame, timestamp_ms)
            
            # Draw results
            output_frame = self.draw_results(processed_frame, signing_status)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                start_time = time.time()
                print(f"FPS: {fps:.1f}")
            
            # Display frame
            annotated_frame = self.draw_landmarks_on_image(processed_frame, results)
            cv2.imshow("annotated frame", annotated_frame)
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset velocity history
                self.left_velocity_history.clear()
                self.right_velocity_history.clear()
                self.left_prev_keypoints = None
                self.right_prev_keypoints = None
                print("Reset velocity history")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Real-time signing detection using webcam')
    parser.add_argument('--threshold', type=float, default=0.01, 
                       help='Velocity threshold for signing detection (default: 0.01)')
    parser.add_argument('--window', type=int, default=5, 
                       help='Moving average window size (default: 5)')
    parser.add_argument('--camera', type=int, required=False,default=0, 
                       help='Camera ID (default: 0)')
    parser.add_argument('--input', type=str, required=False,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, required=False,
                        help='Path to save output annotated video')

    
    args = parser.parse_args()

# Create detector
    detector = RealtimeSigningDetector(
        velocity_threshold=args.threshold,
        moving_average_window=args.window
    )

    if args.input and args.output:
        detector.run_on_video(args.input, args.output)
    else: #Run detection
        detector.run(camera_id=args.camera)
    

if __name__ == "__main__":
    main() 