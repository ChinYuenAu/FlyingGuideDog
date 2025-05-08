from djitellopy import Tello
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import time
import torch

# --------------------------
# Load Models and Deep SORT Trackers
# --------------------------
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=20, n_init=3, max_cosine_distance=0.2)

# Load MiDaS Depth Estimation Model via PyTorch Hub （MiDaS_small for faster inference)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).to(device).eval()     # Load MiDaS model to device and set to eval mode

# Load MiDaS transforms and choose the small_transform (preprocessing pipeline)
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# --------------------------
# Initialize Tello Drone
# --------------------------
tello = Tello()
tello.connect()
tello.streamon()
time.sleep(2)  # Wait for frames to accumulate

# Takeoff and wait for stability
print(f"Battery: {tello.get_battery()}%")
tello.takeoff()
print(f"Height: {tello.get_height()}cm")
tello.send_rc_control(0, 0, 30, 0)                                    # Move up 40 and typically reach at 100cm
tello.send_rc_control(0, 0, 0, 0)

# --------------------------
# Parameters for noise filtering and avoidance
# --------------------------
min_bbox_area = 15000                                                                         # Filter out very small detections
obstacle_speed_threshold = 20                                                                 # Maximum speed in pixels/frame before skipping YOLO tracks
prev_centers = {}                                                                             # Store previous horizontal centers per track
scaling = 0.1                                                                                 # Tuning parameter for lateral avoidance using YOLO/Deep SORT
last_avoid_roll = 0                                                                           # Last sent roll value to the drone
deadband = 2                                                                                  # Deadband threshold for roll control to avoid jittering (e.g., if change is less than 2 units, skip updating)

# Depth-based obstacle threshold - lower values indicates near obstacles
DEPTH_THRESHOLD = 0.25  # Adjust based on calibration
depth_scaling = 100      # Tuning parameter to convert depth error to lateral speed 

# Get ROI bounds for d depth-based obstacle avoidance, left and right regions
def get_roi_bounds(w, h):
    """
    Returns two tuples for left and right ROIs.
    With fraction = 0.3:
    - Left ROI covers from x = 0 to x = 0.3, y = 0.4 * h to y = 0.6 * h 
    - Right ROI covers from x = 0.7 * w to x = w, y = 0.4 * h to y = 0.6 * h
    """
    fraction_x = 0.2
    fraction_y = 0.3
    left_roi = (0, int(h * fraction_y), int(w * fraction_x), int(h * (1 - fraction_y)))
    right_roi = (int(w * (1 - fraction_x)), int(h * fraction_y), w, int(h * (1 - fraction_y)))
    return left_roi, right_roi

# --------------------------
# Main Loop
# --------------------------
try:
    while True:
        # Acquire and resize frame to 640x480
        frame = tello.get_frame_read().frame
        if frame is None or frame.size == 0:                                                  # Check if frame is valid, if not, skip
            continue

        
        frame = cv2.resize(frame, (640, 480))
        h_frame, w_frame, _ = frame.shape
        center_x, center_y = w_frame // 2, h_frame // 2

        # Run YOLO detection and prepare Deep SORT input
        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.5:
                detections.append([x1, y1, x2, y2, conf])

        # Convert detections from [x1, y1, x2, y2, conf] to [[x, y, w, h], conf] to match Deep SORT format
        detections_converted = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            detections_converted.append([[x1, y1, w, h], conf])

        # Update Deep SORT tracker 
        tracks = tracker.update_tracks(detections_converted, frame=frame)

        # -------------------------------
        # Run MiDaS Depth Estimation for lateral obstacle avoidance
        # -------------------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                    # Convert frame (BGR->RGB) 
        input_batch = transform(frame_rgb).to(device)                                         # Apply MiDaS transforms
        with torch.no_grad():                                                                 # Disable gradient calculation(in inference mode, not training, for speed)
            depth_output = midas(input_batch)                                                 # Pass preprocessed image to MiDaS model and get depth output tensor
            # Upsample or resize depth map to original frame size
            depth_map = torch.nn.functional.interpolate(                                      # Resize shape from [batch, height, width] to [batch, 1, height, width] required by interpolation
                depth_output.unsqueeze(1).cpu(),                                              # Force to run on CPU since bicubic interpolation is not supported on MPS.
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()                                                         # Remove batch and channel dimensions                   

        # Normalize and then invert depth map such that near objects ~ 0, far objects ~1
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_map = 1.0 - depth_map

        # Define lateral ROIs - left and right portions 10% of the frame width
        left_roi, right_roi = get_roi_bounds(w_frame, h_frame)
        x1L, y1L, x2L, y2L = left_roi
        x1R, y1R, x2R, y2R = right_roi
    
        # Compute a metric for each ROI. Minimum is too sensitive to noise, while average is too forgiving.
        percentile = 50
        depth_metric_left = np.percentile(depth_map[y1L:y2L, x1L:x2L], percentile)
        depth_metric_right = np.percentile(depth_map[y1R:y2R, x1R:x2R], percentile)
        
        # Visualize depth map for debugging
        depth_disp = (depth_map * 255).astype(np.uint8)

        # Draw ROI boundaries for visualization(yellow boxes)
        cv2.rectangle(depth_disp, (x1L, y1L), (x2L, y2L), (0, 255, 255), 2)
        cv2.rectangle(depth_disp, (x1R, y1R), (x2R, y2R), (0, 255, 255), 2)
        cv2.putText(depth_disp, f"L Depth 50th: {depth_metric_left:.2f}", (x1L + 5, y1L - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  
        cv2.putText(depth_disp, f"R Depth 50th: {depth_metric_right:.2f}", (x1R - 150, y1R - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.rectangle(frame, (x1L, y1L), (x2L, y2L), (0, 255, 255), 2)
        cv2.rectangle(frame, (x1R, y1R), (x2R, y2R), (0, 255, 255), 2)
        cv2.putText(frame, f"L Depth 50th: {depth_metric_left:.2f}", (x1L + 5, y1L - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) 
        cv2.putText(frame, f"R Depth 50th: {depth_metric_right:.2f}", (x1R - 150, y1R - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Height info
        drone_height = tello.get_height() 
        height_text = f"Height: {drone_height} cm" 
        cv2.putText(frame, height_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # -------------------------------
        # Compute YOLO-based lateral avoidance (for dynamic obstacles)
        # -------------------------------
        left_boundary   = int(w_frame * 0.2)
        right_boundary  = int(w_frame * 0.8)
        # top_boundary    = int(h_frame * 0.1)
        # bottom_boundary = int(h_frame * 0.9)
        left_error = 0                                                                        # how much an obstacle on the left intrudes: measured by (x2 - left_boundary)
        right_error = 0                                                                       # how much an obstacle on the right intrudes: measured by (right_boundary - x1)

        # Draw blue boundary lines for left and right
        cv2.line(frame, (left_boundary, 0), (left_boundary, h_frame), (255, 0, 0), 2)
        cv2.line(frame, (right_boundary, 0), (right_boundary, h_frame), (255, 0, 0), 2)

        # Process each track and accumulate avoidance commands
        for track in tracks:
            if not track.is_confirmed():
                continue

            # Convert track bounding box to left, top, right, bottom format
            bbox = track.to_ltrb()
            x1i, y1i, x2i, y2i = map(int, bbox)

            # Skip tracks with too-small bounding boxes (likely noise)
            area = (x2i - x1i) * (y2i - y1i)
            if area < min_bbox_area:
                continue

            # Draw bounding box and track ID for visualization
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f'ID {track.track_id}', (x1i, y1i - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=2)
            

            # Optionally, use obstacle speed filtering
            # If displacement is high, skip this track (it moves fast and won't block the path long)
            obstacle_center_x = (x1i + x2i) / 2
            if track.track_id in prev_centers:
                prev_center = prev_centers[track.track_id]
                displacement = abs(obstacle_center_x - prev_center)                           # Speed as displacement of center between frames
                if displacement > obstacle_speed_threshold:
                    # Update the previous center and skip this track.
                    prev_centers[track.track_id] = obstacle_center_x
                    continue
            prev_centers[track.track_id] = obstacle_center_x                                  # Update (or initialize) the previous center for this track.

            # Determine side by comparing with frame center
            if obstacle_center_x < center_x:
                # Obstacle on the left: we want its right edge to be left of left_boundary to have a clear path
                if x2i > left_boundary:
                    error = x2i - left_boundary
                    left_error = max(left_error, error) 
            else:
                # Obstacle on the right: we want its left edge to be right of right_boundary to have a clear path
                if x1i < right_boundary:
                    error = right_boundary - x1i
                    right_error = max(right_error, error)
   
        # YOLO-based avoidance (if no depth avoidance, consider as fallback)
        if left_error > right_error and left_error > 0:
            avoid_roll_yolo = int(scaling * left_error)                                       # There is a significant obstacle on the left. Move right (positive roll)
        elif right_error > left_error and right_error > 0:
            avoid_roll_yolo = -int(scaling * right_error)                                     # There is a significant obstacle on the right. Move left (negative roll)
        else:
            avoid_roll_yolo = 0                                                               # No significant obstacles on either side, so no lateral movement.

        # -------------------------------
        # Compute Depth-based Avoidance 
        # -------------------------------
        if depth_metric_right < DEPTH_THRESHOLD:                                              # e.g. 0.1 < 0.3 ==> -(0.2-0.1)*100 = -10 
            depth_error_right = DEPTH_THRESHOLD - depth_metric_right
            avoid_roll_depth = -int(depth_error_right * depth_scaling)                        # Move left (negative roll)
        else:
            avoid_roll_depth = 0                                                              

        if depth_metric_left < DEPTH_THRESHOLD:
            depth_error_left = DEPTH_THRESHOLD - depth_metric_left
            avoid_roll_depth = int(depth_error_left * depth_scaling)                          # Move right (positive roll)

        if (depth_metric_left < DEPTH_THRESHOLD) and (depth_metric_right < DEPTH_THRESHOLD):
            # Use the side with the larger depth error
            if depth_error_left > depth_error_right:
                avoid_roll_depth = int(depth_error_left * depth_scaling)                       # Move right (positive roll)
            else:
                avoid_roll_depth = -int(depth_error_right * depth_scaling)                      # Move left (negative roll)

        # Emphasize depth-based avoidance if it is significant, override YOLO-based avoidance
        if avoid_roll_depth != 0:
            avoid_roll = avoid_roll_depth
            print("Avoidance: Depth-based")     
        else:
            avoid_roll = avoid_roll_yolo
            print(f"Avoidance: YOLO-based, L error={left_error}, R error={right_error}")

        if abs(avoid_roll) < 10:
            forward_speed = 15
        else:
            forward_speed = 10
        
        # Debut output
        if avoid_roll != 0:
            print(f"Avoidance: roll={avoid_roll} (Depth LR: L={depth_metric_left:.2f}, R={depth_metric_right:.2f}; YOLO L error={left_error}, R error={right_error})")
        else:
            print("Path clear - moving forward.")

        # Show results for debugging
        combined_frame = np.hstack((frame, cv2.cvtColor(depth_disp, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Combined View", combined_frame)


        # Apply deadband to avoid oscillations
        if abs(avoid_roll - last_avoid_roll) < deadband:
            avoid_roll = last_avoid_roll                                                      # Use the previous roll value
        else:
            last_avoid_roll = avoid_roll                                                      # Update the stored roll value

        # Send control command: (lr, fw, ud, yaw)
        tello.send_rc_control(avoid_roll, forward_speed, 0, 0)
        print(f"Control: lr={avoid_roll}, fw={forward_speed}, ud=0, yaw=0")

        key = cv2.waitKey(20) & 0xFF
        print("Key pressed:", key)
        if key == ord('q'):
            for _ in range(3):
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.05)  # short pause between commands
            print("Emergency stop initiated. Preparing to land...")
            break
 
finally:
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamoff()
    cv2.destroyAllWindows()
    tracker = None
    model = None    
    tello.end()
