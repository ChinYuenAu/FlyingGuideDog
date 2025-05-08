import time
import cv2
import numpy as np
import math
import torch
from djitellopy import Tello
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------------
# Camera parameters
# --------------------------
Camera_fx = 916.4798394056434                                                                 # Focal lengths in pixels (how much the lens magnifies the image)
Camera_fy = 919.9110948929223                                                                           
Camera_cx = 483.57407010014435                                                                # Principal point (the optical center of the camera, usually near the center of the image)
Camera_cy = 370.87084181752994

# Distortion coefficients
Camera_k1 = 0.08883662811988326                                                               # Radial distortion (causes straight lines to appear curved)  
Camera_k2 = -1.2017058559646074                                                               # Large negative distortion, likely causing "barrel distortion" 
Camera_k3 = 4.487621066094839
Camera_p1 = -0.0018395141258008667                                                            # Tangential distortion (causes the image to look tilted or skewed)     
Camera_p2 = 0.0015771769902803328    

# Forming the camera matrix and distortion coefficients
camera_matrix = np.array([[Camera_fx, 0, Camera_cx],
                          [0, Camera_fy, Camera_cy], 
                          [0, 0, 1]], dtype="double")
dist_coeffs = np.array([Camera_k1, Camera_k2, Camera_p1, Camera_p2, Camera_k3])

# --------------------------
# Load Models and Deep SORT Trackers
# --------------------------
model   = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=20, n_init=3, max_cosine_distance=0.2)

# Load MiDaS Depth Estimation Model via PyTorch Hub （MiDaS_small for faster inference)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).to(device).eval()   # Load MiDaS model to device and set to eval mode
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform                   # Preprocessing pipeline for MiDaS_small model
prev_obs_centers = {}                                                                         # Dictionary to store previous obstacle centers for filtering fast-moving objects

# ---------------------------
# Detfine L/R depth ROI bounds for MiDaS
# ----------------------------
def get_roi_bounds(w, h):
    """
    Returns two tuples for left and right ROIs.
    """
    fraction_x = 0.2
    fraction_y = 0.3
    left_roi = (0, int(h * fraction_y),                                                       # Left ROI covers from x = 0 to x = 0.2 * w, y = 0.3 * h to y = 0.7 * h 
                 int(w * fraction_x), int(h * (1 - fraction_y)))      
    right_roi = (int(w * (1 - fraction_x)), int(h * fraction_y),                              # Right ROI covers from x = 0.8 * w to x = w, y = 0.3 * h to y = 0.7 * h
                 w, int(h * (1 - fraction_y))) 

    return left_roi, right_roi

# ----------------------------
# Initialize Tello 
# ----------------------------
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamoff()
tello.streamon()
frame_reader = tello.get_frame_read()                                                        # Get the frame reader object

# wait until the first frame is available
while (frame_reader.frame is None) or (frame_reader.frame.size == 0):
    time.sleep(0.1)                                              
print("First frame received.")                            

# ------------------------
# ArUco marker tracking setup
# ------------------------
# Get the predefined dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)                        # retrieve predifined dictionary of 6x6 Aruco markers with 250 unique IDs 
aruco_params = cv2.aruco.DetectorParameters()                                                 # create a set of parameters for marker detection (Thresholds for binarization, Minimum/maximum marker sizes, Corner refinement settings)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)  

# ------------------------
# PID and tracking parameters
# ------------------------
pid_x = [0.2, 0, 0.02]                                                                        # Proportional gain→Reacts to current error, Integral→Accumulates past errors (set to 0 here), Derivative gain→Increases if drone overshoots or oscillates too much.
pid_y = [0.1, 0, 0.02]                                                                        # PID gains for vertical control
# Previous errors: used to calculate the derivative term for smooth PID-based tracking
pError_x, pError_y = 0, 0                                                                     # For yaw control (horizontal movement) and vertical control (up/down movement)  
pError_area = 0                                                                               # For forward/backward control (distance from marker)
prev_fb = 0                                                                                   # Previous forward/backward speed     
marker_width_cm = 9.4                                                                         # Adjust to your marker's actual width(cm)
distance_min = 90.0                                                                           # Adjust based on testing: Min distance from marker to drone (backward movement if too close)
distance_max = 100.0                                                                          # Adjust based on testing: Max distance from marker to drone (forward movement if too far)
area_max = ((marker_width_cm * Camera_fx) / distance_min) ** 2                                # Max marker area in pixels (backward movement if too close)
area_min = ((marker_width_cm * Camera_fx) / distance_max) ** 2                                # Min marker area in pixels (forward movement if too far)
fbRange = [area_min, area_max]                                                                # Range of marker areas in pixels
# fbRange = [4000, 6000] (Original)
# distance(cm) = (marker_width(cm) * focal_length(px)) / sqrt(marker_area)
# marker_area(pixel) = (marker_width(cm) * focal_length(px)) / distance(cm) ** 2 
# (9.4 * 916.4798394056434) / sqrt(46380) = 40.00  || (9.4 * 916.4798394056434) / sqrt(36650) = 45.00  || (9.4 * 916.4798394056434) / sqrt(29680) = 50.00 
# (9.4 * 916.4798394056434) / sqrt(24530) = 55.00  || (9.4 * 916.4798394056434) / sqrt(20610) = 60.00   
frame = None
# For marker memory
last_marker_direction = None
last_marker_position = None
last_marker_time = 0
marker_memory_timeout = 5.0                                                                   # Duration in seconds to remember marker direction and position in search mode
last_avoid_roll = 0                                                                           # Last roll command for obstacle avoidance

def track_aruco_marker(frame):
    """
    Track ArUco marker in video feed and update tracking info.

    Parameters:
        frame : Current video frame from the drone's camera

    Returns:
        marker_found : Boolean indicating if the marker was found
        (lr, fb, ud, yaw) : Control commands for left/right, forward/backward, up/down, and yaw
    """

    global pError_x, pError_y, last_marker_position, last_marker_direction, last_marker_time, marker_distance, marker_pixel_width

    marker_found = False
    lr = fb = ud = yaw = 0    
    if frame is None:
        return marker_found, (lr, fb, ud, yaw)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect ArUco markers
    # corners: list of marker corners detected in the image, each marker is (1,4,2) 
    # ids: unique IDs of the detected markers
    # rejected: markers detected but didn't meet certain criteria (like size, orientation, etc.)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)                           

    if ids is not None and len(corners) > 0:
        marker_found = True                                                                   # Marker detected
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)                                    # Draw the detected marker
        corner = corners[0][0]                                                                # Four corners coordinates of the first (and only) marker

        # Define real-world 3D marker coordinates (assuming square, centered at (0,0,0))
        marker_3d_points = np.array([
            [-marker_width_cm / 2, marker_width_cm / 2, 0],                                   # Top-left
            [marker_width_cm / 2, marker_width_cm / 2, 0],                                       # Top-right
            [marker_width_cm / 2, -marker_width_cm / 2, 0],                                      # Bottom-right
            [-marker_width_cm / 2, -marker_width_cm / 2, 0]                                   # Bottom-left
            ], dtype=np.float32)

        # Solve PnP problem to estimate marker distance
        success, rvec, tvec = cv2.solvePnP(marker_3d_points, corner, camera_matrix, dist_coeffs)

        if success:
            # Store distance from camera to marker in cm
            marker_distance = float(tvec[2][0])                                               
            cv2.putText(frame, f"Marker Distance: {marker_distance:.2f} cm", 
                        (int(corner[0][0]), int(corner[0][1]) - 15), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate marker pixel in pixels (using corners 0 and 1) 
            marker_pixel_width = np.linalg.norm(corner[0] - corner[1])                         

            # # Store pixels per cm ratio 
            # pixels_per_cm_at_marker = marker_pixel_width / marker_width_cm                    
            
            # cv2.putText(frame, f"Pixels per cm: {pixels_per_cm_at_marker:.2f}",
            #             (int(corner[0][0]), int(corner[0][1]) - 30),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        # Bounding box & center calculation
        x, y, w, h = cv2.boundingRect(corner)                        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)                          # Draw the bounding box of the marker                                                         
        area = w * h                                                                          # Calculate area and center
        
        # Compute center of marker   
        center_x = int((corner[:, 0].sum()) / 4)
        center_y = int((corner[:, 1].sum()) / 4)

        # Update previous position for tracking
        current_time = time.time()
        if last_marker_position is not None:
            dx = center_x - last_marker_position[0]
            dy = center_y - last_marker_position[1]
            last_marker_direction = (dx, dy)                                                  # Calculate direction of marker based on previous position 
        else:
            last_marker_position = (center_x, center_y)

        last_marker_time = current_time

        # Tracking logic here
        pError_x, pError_y, lr, fb, ud, yaw = trackObj((center_x, center_y), area, 
                                      frame.shape[0], frame.shape[1], 
                                      pid_x, pid_y, pError_x, pError_y)

    return marker_found, (lr, fb, ud, yaw)                                                        # Return marker found status and tracking info


def trackObj(marker_center, area, frame_height, frame_width, pid_x, pid_y, pError_x, pError_y, dt=0.1):
    """
    PD Controller for marker centering and distance control.

    Parameters:
        marker_center : Tuple (x, y) representing the marker center in pixels
        area          : Area of the detected marker in pixels
        frame_width   : Width of the video frame in pixels
        pid_x         : List of PID gains [Kp, Kd] for yaw rotation (Integral not used here)
        pid_y         : List of PID gains [Kp, Kd] for vertical control
        pError_x      : Previous horizontal error (for yaw derivative term)
        pError_y      : Previous vertical error (for up/down derivative term)
        dt            : Time interval between frames/control updates

    Returns:
        new_pError_x, new_pError_y : Updated previous errors for the next iteration
        lr, fb, ud, yaw            : Control commands for left/right, forward/backward, up/down, and yaw
    """ 
    global prev_fb, pError_area
    # unpack marker center coordinates
    x, y = marker_center
    
    # ----- Horizontal (Yaw) control -----
    error_x = x - (frame_width // 2)                                                          # Error between the center of the marker and the center of the frame
    dError_x = (error_x - pError_x) / dt                                                      # derivative of horizontal error over time            
    yaw = pid_x[0] * error_x + pid_x[2] * dError_x                                            # PD control for yaw rotation                                                         
    yaw = int(np.clip(yaw, -80, 80))                                                          # Limits yaw speed between -100 (full left) and 100 (full right) to prevent overcorrection

    # ----- Left/Right control -----
    # Allows to drone to fly directly toward the marker rather than along a curved path
    k_lr = 0.1  # Tune this value     
    lr = k_lr * error_x                                            
    lr = int(np.clip(lr, -20, 20))                                                    

    # ----- Forward/Backward control -----
    # Define target area (desired marker size in pixels)
    target_area = (fbRange[0] + fbRange[1]) / 2
    error_area = target_area - area
    dError_area = (error_area - pError_area) / dt                                             # derivative of area error over time         
    # Proportional gain for fb (tune this value to adjust the drone's speed)
    # k_fb = 0.0008  
    k_fb = 0.001              
    # Derivative gain for fb (increase this value to reduce oscillations by reacting to the rate of change of the error)                                                                
    # k_fb_d = 0.001                         
    k_fb_d = 0.0012    
    fb_dynamic = k_fb * error_area + k_fb_d * dError_area                                     # Dynamic forward/backward speed based on area error
    pError_area = error_area                                                                  # Update previous area error
    
    # Set a minimum speed threshold
    min_speed = 10
    if area < fbRange[0]:
        # Drone is too far; error_area is positive so fb_dynamic should be positive; Ensure a minimum forward speed.
        fb = fb_dynamic if abs(fb_dynamic) >= min_speed else min_speed
    elif area > fbRange[1]:
        # Drone is too close; error_area is negative so fb_dynamic should be negative; Ensure a minimum backward speed.
        fb = fb_dynamic if abs(fb_dynamic) >= min_speed else -min_speed
    else:
        # Within the acceptable range
        fb = 0

    # Apply low-pass filter to smooth the command
    alpha = 0.4                                                                               # Smoothing factor (adjust as needed between 0 and 1); lower value give more smoothing
    fb = alpha * fb + (1 - alpha) * prev_fb                            
    prev_fb = fb
    fb = int(np.clip(fb, -100, 100))                                                          # Clip the speed to the Tello's limits (typically -100 to 100)

    # ----- Vertical (Up/Down) control -----
    # Use the frame's height (global frame variable) for vertical error
    error_y = (frame_height // 2) - y                                                         # Positive error if marker is above the center, negative if below; In OpenCV, the y-axis increases downwards
    dError_y = (error_y - pError_y) / dt                                                      # derivative of vertical error over time
    ud = pid_y[0] * error_y + pid_y[2] * dError_y                                             # PD control for up/down movement
    ud = int(np.clip(ud, -50, 50))                                                            # Clip the speed to the Tello's limits (typically -100 to 100)

    # if marker is not detected, stop yaw rotation
    if x == 0:
        yaw = 0
        error = 0

    # Command format: left/right, forward/backward, up/down, yaw)
    # return the current errors and commands                         
    return error_x, error_y, lr, fb, ud, yaw                                                



def detect_obstacle(frame, model, tracker, midas, transform, device, prev_obs_centers,
                    yolo_scaling=0.1, depth_threshold=0.25, depth_scaling=100, obstacle_speed_threshold=20, min_bbox_area=15000, deadband=2):
    """
    Detect obstacles using YOLO, DeepSort, and MiDaS depth estimation.
    Returns a lateral command: negative = move left, positive = move right, zero = clear path.
    
    Parameters:
        frame                    : Current video frame from the drone's camera
        model                    : YOLO model for object detection
        tracker                  : DeepSort tracker for tracking detected objects
        midas                    : MiDaS model for depth estimation
        transform                : Preprocessing pipeline for MiDaS
        device                   : Device to run the models on (CPU/MPS/GPU)
        prev_obs_centers         : Dictionary storing previous obstacle centers to filter fast-moving objects
        yolo_scaling             : Scaling factor for YOLO-based lateral avoidance
        depth_threshold          : Threshold for depth estimation to detect obstacles
        depth_scaling            : Scaling factor for depth-based avoidance
        obstacle_speed_threshold : Speed threshold to filter fast-moving objects
        min_bbox_area            : Minimum bounding box area to filter very small objects
        deadband                 : Deadband to avoid oscillations in avoidance commands

    Returns:
        avoid_roll : Lateral command for obstacle avoidance
    """

    global last_avoid_roll

    h_frame, w_frame, _ = frame.shape
    center_x, center_y = w_frame // 2, h_frame // 2

    # Run YOLO detection and prepare Deep SORT input
    results = model(frame)[0]

    print("YOLO device:", next(model.parameters()).device)                                    # It will show "mps:0" if it is MPS, and "cpu" if it is CPU.

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        if conf > 0.5:
            detections.append([x1, y1, x2, y2, conf])                         # [[x, y, w, h], conf] to match Deep SORT format
    
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
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                        # Convert frame (BGR->RGB) 
    input_batch = transform(frame_rgb).to(device)                                             # Apply MiDaS transforms
    with torch.no_grad():                                                                     # Disable gradient calculation(in inference mode, not training, for speed)
        depth_output = midas(input_batch)                                                     # Pass preprocessed image to MiDaS model and get depth output tensor
        # Upsample or resize depth map to original frame size
        depth_map = torch.nn.functional.interpolate(                                          # Resize shape from [batch, height, width] to [batch, 1, height, width] required by interpolation
            depth_output.unsqueeze(1).cpu(),                                                  # Force to run on CPU since bicubic interpolation is not supported on MPS.
            size=(h_frame, w_frame),                                                          # Resize to original frame size
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()                                                             # Remove batch and channel dimensions                   

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

    # Exponential moving average smooothing for depth metrics
    alpha = 0.3         # 0 < alpha < 1, lower = smoother
    if 'prev_dl' not in detect_obstacle.__dict__:
        detect_obstacle.prev_dl = depth_metric_left
        detect_obstacle.prev_dr = depth_metric_right
    depth_metric_left = alpha * depth_metric_left + (1 -alpha) * detect_obstacle.prev_dl
    depth_metric_right = alpha * depth_metric_right + (1 -alpha) * detect_obstacle.prev_dr
    detect_obstacle.prev_dl, detect_obstacle.prev_dr = depth_metric_left, depth_metric_right

    # Visualize depth map for debugging
    depth_disp = (depth_map * 255).astype(np.uint8)

    # Draw ROI boundaries for visualization(yellow boxes)
    cv2.rectangle(depth_disp, (x1L, y1L), (x2L, y2L), (0, 0, 255), 2)                       # Left ROI
    cv2.rectangle(depth_disp, (x1R, y1R), (x2R, y2R), (0, 0, 255), 2)                       # Right ROI
    cv2.putText(depth_disp, f"L Depth 50th: {depth_metric_left:.2f}", (x1L + 5, y1L - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  
    cv2.putText(depth_disp, f"R Depth 50th: {depth_metric_right:.2f}", (x1R - 130, y1R - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.rectangle(frame, (x1L, y1L), (x2L, y2L), (0, 0, 255), 2)
    cv2.rectangle(frame, (x1R, y1R), (x2R, y2R), (0, 0, 255), 2)
    cv2.putText(frame, f"L Depth 50th: {depth_metric_left:.2f}", (x1L + 5, y1L - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
    cv2.putText(frame, f"R Depth 50th: {depth_metric_right:.2f}", (x1R - 130, y1R - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Height, battery and speed info
    drone_height = tello.get_height()  
    height_text = f"Height: {drone_height} cm" if drone_height != -1 else "Height: N/A" 
    cv2.putText(frame, height_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    battery_text = f"Battery: {tello.get_battery()}%" if tello.get_battery() != -1 else "Battery: N/A"
    cv2.putText(frame, battery_text, (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) 
    # speed_text = f"X-Asis Speed: {tello.get_speed_x()} cm/s" if tello.get_speed_x() != -1 else "Speed: N/A"
    # cv2.putText(frame, speed_text, (10, h_frame - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # -------------------------------
    # Compute YOLO-based lateral avoidance (for dynamic obstacles)
    # -------------------------------
    left_boundary   = int(w_frame * 0.2)
    right_boundary  = int(w_frame * 0.8)
    # top_boundary    = int(h_frame * 0.1)
    # bottom_boundary = int(h_frame * 0.9)
    left_error = right_error = 0                                                              # how much an obstacle on the left/right intrudes: measured by (x2 - left_boundary) or (right_boundary - x1)

    # # Draw blue boundary lines for left and right
    # cv2.line(frame, (left_boundary, 0), (left_boundary, h_frame), (255, 0, 0), 2)
    # cv2.line(frame, (right_boundary, 0), (right_boundary, h_frame), (255, 0, 0), 2)

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

        # Filter out fast moving objects (likely not obstacles)
        obstacle_center_x = (x1i + x2i) / 2
        if track.track_id in prev_obs_centers:
            prev_center = prev_obs_centers[track.track_id]
            displacement = abs(obstacle_center_x - prev_center)                               # Speed as displacement of center between frames
            if displacement > obstacle_speed_threshold:
                # Update the previous center and skip this track.
                prev_obs_centers[track.track_id] = obstacle_center_x
                continue
        prev_obs_centers[track.track_id] = obstacle_center_x                                  # Update (or initialize) the previous center for this track.

        # Draw bounding box and track ID for visualization
        # cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color=(0, 255, 0), thickness=2)
        # cv2.putText(frame, f'ID {track.track_id}', (x1i, y1i - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=2)
      
        # Determine side by comparing with frame center
        if obstacle_center_x < center_x and x2i > left_boundary:                              # Obstacle on the left: we want its right edge to be left of left_boundary to have a clear path
            left_error = max(left_error, x2i - left_boundary)
        elif obstacle_center_x > center_x and x1i < right_boundary:                           # Obstacle on the right: we want its left edge to be right of right_boundary to have a clear path
            right_error = max(right_error, right_boundary - x1i)
    
    # Add decay so the error returns to 0 once nothing overlaps
    if left_error == 0 and right_error == 0:        # no boxes in either zone
        left_error  = getattr(detect_obstacle, 'left_err_prev', 0)  * 0.7
        right_error = getattr(detect_obstacle, 'right_err_prev', 0) * 0.7

    # Remember last values for next frame
    detect_obstacle.left_err_prev  = left_error
    detect_obstacle.right_err_prev = right_error

    # Calculate avoidance based on YOLO
    avoid_roll_yolo = 0
    if left_error > right_error and left_error > 0:
            avoid_roll_yolo = int(yolo_scaling * left_error)                                  # There is a significant obstacle on the left. Move right (positive roll)
    elif right_error > left_error and right_error > 0:
        avoid_roll_yolo = -int(yolo_scaling * right_error)                                    # There is a significant obstacle on the right. Move left (negative roll)

    # Depth-based avoidance overrides YOLO if significant
    avoid_roll_depth = 0
    if depth_metric_right < depth_threshold:                                                  # e.g. 0.1 < 0.3 ==> -(0.2-0.1)*100 = -10 
        depth_error_right = depth_threshold - depth_metric_right
        avoid_roll_depth = -int(depth_error_right * depth_scaling)                            # Move left (negative roll)
    if depth_metric_left < depth_threshold:
        depth_error_left = depth_threshold - depth_metric_left
        avoid_roll_depth = int(depth_error_left * depth_scaling)                              # Move right (positive roll)

    # Use larger depth error if both sides are below threshold
    if (depth_metric_left < depth_threshold) and (depth_metric_right < depth_threshold):
        avoid_roll_depth = int((depth_error_left - depth_error_right) * depth_scaling)      # Move left or right based on the difference

    # Emphasize depth-based avoidance over YOLO
    # if avoid_roll_depth != 0:
    #     avoid_roll = avoid_roll_depth
    #     print("Avoidance: Depth-based")
    # else:
    #     avoid_roll = avoid_roll_yolo
    #     print(f"Avoidance: YOLO-based, L error={left_error}, R error={right_error}")
    w_depth = 1
    w_yolo = 0
    avoid_roll = int(w_depth * avoid_roll_depth) + int(w_yolo * avoid_roll_yolo)               # Combine depth and YOLO avoidance commands
    print(f"Avoidance fusion: Depth: {avoid_roll_depth}, YOLO: {avoid_roll_yolo} => roll= {avoid_roll}")

    # Apply deadband to avoid oscillations
    if abs(avoid_roll) < 1:
        avoid_roll = 0                                                                        # No avoidance if the command is too small
    elif abs(avoid_roll - last_avoid_roll) < deadband:
        avoid_roll = last_avoid_roll                                                          # Use the previous roll value
    else:
        last_avoid_roll = avoid_roll                                                          # Update the stored roll value
    
    # Track how many consecutive frames we issued no avoidance
    if avoid_roll == 0:
        # detect_obstacle.zero_frames = detect_obstacle.get('zero_frames', 0) + 1
        detect_obstacle.zero_frames = getattr(detect_obstacle, 'zero_frames', 0) + 1
    else:
        detect_obstacle.zero_frames = 0

    # If we have been clear for >50 frames (~7 seconds at 7fps), wipe old track centres    
    if detect_obstacle.zero_frames > 50:
        prev_obs_centers.clear()
        detect_obstacle.zero_frames = 0

    # Print avoidance information
    if avoid_roll != 0:
        print(f"Avoidance: roll={avoid_roll} (Depth LR: L={depth_metric_left:.2f}, R={depth_metric_right:.2f}; YOLO L error={left_error}, R error={right_error})")
    else:
        print("Path clear - moving forward.")

    # Show the depth map and the original frame side by side
    combined_frame = np.hstack((frame, cv2.cvtColor(depth_disp, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Combined View", combined_frame)

    return int(np.clip(avoid_roll, -30, 30))                                                  # Clip the roll command to (-30 to 30)


def search_with_memory(last_marker_position, last_marker_direction, last_marker_time, timeout):
    """
    Handle recovery when the marker is not found with its last known position and direction.

    Params:
        last_marker_position : The last known position of the marker.
        last_marker_direction: The last known direction of the marker.
        last_marker_time     : The time when the marker was last seen.
        timeout              : The timeout duration for marker recovery.

    Returns:
        A tuple containing the speed commands for left/right, forward/backward, up/down, and yaw.
    """

    current_time = time.time()
    print(f"Last marker position: {last_marker_position}, Last marker direction: {last_marker_direction}")
    if last_marker_position is not None and (current_time - last_marker_time) < timeout:
        if last_marker_direction is not None:
            dx, dy = last_marker_direction
            yaw_speed = int(np.clip(dx * 0.2, -30, 30))                                       # Scale x-direction to yaw speed
            ud_speed = int(np.clip(dy * 0.1, -30, 30))                                        # Scale y-direction to up/down speed   
            lr_speed = int(np.clip(dx * 0.1, -30, 30))                                        # Scale x-direction to left/right speed (drifting)
            fb_speed = 10
            return (lr_speed, fb_speed, ud_speed, yaw_speed)
        else:
            print("No direction available.")
            return (0, 0, 0, 20)
    else:
        print("No memory available.")
        return (0, 0, 0, 20)

# -----------------------------
# Main loop (No multithreading)
# -----------------------------
flying = False                                                                                # Keep track of whether the drone is airborne
quit_program = False

def main():
    global flying, quit_program, last_marker_direction, last_marker_position, last_marker_time

    try:
        while True:
            lr = fb = ud = yaw = 0
            avoid_roll = 0

            # Get the current frame from Tello
            frame = frame_reader.frame
            if frame is None or frame.size == 0:                                             # rare blank frames, skip
                continue                                                                      
            # Correct the frame for camera distortion
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            frame = cv2.resize(frame, (640, 480))                

            if flying:
                # Track ArUco Marker
                marker_found, (lr, fb_marker, ud, yaw) = track_aruco_marker(frame)

                # Detect obstacles and get roll command
                avoid_roll = detect_obstacle(frame, model, tracker, midas, transform, device, prev_obs_centers)

                # Merge logic: prioritize obstacle avoidance over marker tracking
                if avoid_roll != 0:
                    # If obstacle avoidance is triggered, send roll command
                    print("Obstacle detected! Avoiding...")
                    fb_avoid = max(10, int(0.5 * fb_marker))                                 # Adjust forward speed based on marker tracking
                    rc = (avoid_roll, fb_avoid, 0, 0)
                elif marker_found:
                    print("Marker found! Tracking...")
                    rc = (int(lr), int(fb_marker), int(ud), int(yaw))
                else:
                    # If no marker is found, search with memory
                    print("Marker not found! Searching...")
                    rc = search_with_memory(last_marker_position, last_marker_direction, last_marker_time, marker_memory_timeout)
                
                # Send RC control commands to Tello
                tello.send_rc_control(*rc)
                print("rc command format: left/right, forward/backward, up/down, yaw)")
            
            else:
                # Idle preview
                idle = frame.copy()
                # Display the start frame if the drone is not airborne
                cv2.putText(idle, "Press 's' to takeoff", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Idle Preview", idle)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not flying:
                cv2.destroyWindow("Idle Preview")
                tello.takeoff()
                tello.send_rc_control(0, 0, 40, 0)                                            # Ascend for 40 cm
                time.sleep(2)                                                                 # Wait for Tello to stabilize
                flying = True
            elif key == ord('l') and flying:
                tello.land()
                flying = False
            elif key == ord('q'):
                quit_program = True
                break
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except cv2.error as cv:
        print(f"OpenCV error: {cv}")
    except Exception as e:
        print(f"General error: {e}")

    finally:
        if flying or quit_program:
            try:
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                print("Drone landing!")
            except Exception as e:
                print("Error sending land command: ", e)
        try:
            print(f"Battery: {tello.get_battery()}%")
            tello.streamoff()
            cv2.destroyAllWindows()                                                               # Close all OpenCV windows
            tello.end()
        except Exception as e:
            print("Error ending Tello connection: ", e)


if __name__ == "__main__":
    main()  