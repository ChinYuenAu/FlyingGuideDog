from djitellopy import Tello
import cv2
import numpy as np
import pygame
import os
import json
import time
import random
import math
from ultralytics import YOLO

# --------------------------
# Load YOLO model
# --------------------------
model = YOLO('yolov8n.pt')                                                                    # Load the YOLOv8 model

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
# Initialize Tello 
# --------------------------
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.for_back_velocity = 0
tello.left_right_velocity = 0
tello.up_down_velocity = 0
tello.yaw_velocity = 0
tello.speed = 0

# Reset any previous stream and start a new one
tello.streamoff()
tello.streamon()

# ------------------------
# ArUco marker tracking setup
# ------------------------
# Get the predefined dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)                        # retrieve predifined dictionary of 6x6 Aruco markers with 250 unique IDs 
aruco_params = cv2.aruco.DetectorParameters()                                                 # create a set of parameters for marker detection (Thresholds for binarization, Minimum/maximum marker sizes, Corner refinement settings)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)  
frameWidth = 360  # or the actual width of your video frame  

# ------------------------
# PID and tracking parameters
# ------------------------
pid_x = [0.2, 0, 0.02]                                                                        # Proportional gain→Reacts to current error, Integral→Accumulates past errors (set to 0 here), Derivative gain→Increases if drone overshoots or oscillates too much.
pid_y = [0.1, 0, 0.02]                                                                        # PID gains for vertical control
# Previous errors: used to calculate the derivative term for smooth PID-based tracking
pError_x = 0                                                                                  # For yaw control (horizontal movement)
pError_y = 0                                                                                  # For vertical control (up/down movement) 
pError_area = 0                                                                               # For forward/backward control (distance from marker)
prev_fb = 0                                                                                   # Previous forward/backward speed     
marker_width_cm = 9.4                                                                         # Adjust to your marker's actual width(cm)
distance_min = 45.0                                                                           # Adjust based on testing: Min distance from marker to drone (backward movement if too close)
distance_max = 50.0                                                                           # Adjust based on testing: Max distance from marker to drone (forward movement if too far)
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
marker_memory_timeout = 3.0                                                                   # Duration in seconds to remember marker direction and position in search mode
obstacle_history = []                                                                         # Store obstacle history for temporal consistency in obstacle detection


def track_aruco_marker():
    """
    Track ArUco marker in video feed, estimate the marker's distance from the drone and update its last position for tracking.
    """
    global pError_x, pError_y, frame, last_marker_position, last_marker_direction, last_marker_time, marker_distance, marker_pixel_width

    marker_found = False
    if frame is None:
        return marker_found

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

            # Store pixels per cm ratio 
            pixels_per_cm_at_marker = marker_pixel_width / marker_width_cm                    
            
            cv2.putText(frame, f"Pixels per cm: {pixels_per_cm_at_marker:.2f}",
                        (int(corner[0][0]), int(corner[0][1]) - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        # Bounding box & center calculation
        x, y, w, h = cv2.boundingRect(corner)                        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)                          # Draw the bounding box of the marker                                                         
        area = w * h                                                                          # Calculate area and center
        cv2.putText(frame, f"Marker Area: {area}", 
                    (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
        pError_x, pError_y = trackObj(tello, (center_x, center_y), area, 
                                      frame.shape[0], frame.shape[1], 
                                      pid_x, pid_y, pError_x, pError_y)

    return marker_found


def trackObj(tello, marker_center, area, frame_height, frame_width, pid_x, pid_y, pError_x, pError_y, dt=0.1):
    """
    Control yaw, left/right, front/back, up/down movement using PD controllers based on the marker's position and area.

    Parameters:
        tello         : Tello drone object
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
    """ 
    global prev_fb, pError_area
    # fb = 0                                                                                  # forward-backward movement
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
    alpha = 0.1                                                                               # Smoothing factor (adjust as needed between 0 and 1); lower value give more smoothing
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

    # ----- Send Combined RC Commands -----
    # Command format: left/right, forward/backward, up/down, yaw)
    tello.send_rc_control(lr, fb, ud, yaw)    

    # return the current error so it can be used in the next cycle as pError                                         
    return error_x, error_y                                                


def detect_obstacle(frame):
    """
    To be implemented using YOLOv8.
    """
    obstacles = {}
    results = model(frame)

    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates and confidence score
            x1, y1, x2, y2 = box.xyxy[0].tolist()                                                 
            confidence = box.conf[0].item()                                                   # Confidence score of the detection
            class_id = int(box.cls[0].item*())                                                # Class ID of the detected object

            # Compute center coordinates and area 
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            bbox_width = int(x2 - x1)
            bbox_height = int(y2 - y1)
            area = bbox_width * bbox_height

            # Estimate distance: Use a known object width (in cm) and the focal length (Camera_fx)
            known_width_cm = 50  # Example known width; adjust based on expected obstacle sizes
            if bbox_width <= 0:
                continue
            distance_cm = (known_width_cm * Camera_fx) / bbox_width
            
            # Compute path proximity: Euclidean distance from the frame center
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            path_proximity = np.linalg.norm(np.array([center_x, center_y]) - np.array(frame_center))
            
            obstacles[(center_x, center_y)] = {
                'area': area,
                'distance_cm': distance_cm,
                'confidence_score': confidence,
                'path_proximity': path_proximity
            }
    
    # Update temporal history for obstacle consistency
    update_temporal_history(obstacles)
    
    return obstacles



# Persistent storage for obstacle tracking
obstacle_history = {}  # {id: {center: (cx, cy), area_history: [...], initial_area: A0, initial_marker_distance: D0, ...}}
next_obj_id = 0        # incremental counter for assigning IDs if no tracker ID is available

def detect_obstacle(frame, marker_distance):
    global obstacle_history, next_obj_id
    
    detections = model.detect(frame)  # Pseudocode: get YOLO detections for this frame
    detected_obstacles = []  # list to hold results for this frame
    
    # Parameters for matching and smoothing
    center_distance_threshold = 50  # pixel threshold to match same object (tune as needed)
    N_HISTORY = 5                   # how many past frames to keep for smoothing
    
    for det in detections:
        x, y, w, h = det['bbox']  # bounding box coordinates from YOLO
        cx = x + w/2
        cy = y + h/2
        area = w * h
        label = det.get('class', None)
        
        # Determine an identifier for this obstacle
        if 'id' in det:
            obj_id = det['id']  # use tracker-provided ID if available
        else:
            obj_id = None
            # Try to match by center proximity to an existing obstacle
            for oid, info in obstacle_history.items():
                prev_cx, prev_cy = info['center']
                dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2) ** 0.5
                if dist < center_distance_threshold:
                    obj_id = oid
                    break
            # If not matched, assign a new ID
            if obj_id is None:
                obj_id = f"obj_{next_obj_id}"
                next_obj_id += 1
        
        # Fetch or create history entry for this obstacle
        if obj_id not in obstacle_history:
            # New obstacle: initialize its history entry
            obstacle_history[obj_id] = {
                'center': (cx, cy),
                'area_history': [],
                'initial_area': area,
                'initial_marker_distance': marker_distance,
                'label': label
            }
        info = obstacle_history[obj_id]
        
        # Update the center position
        info['center'] = (cx, cy)
        info['label'] = label
        # Update area history (append current area and trim to max length)
        info['area_history'].append(area)
        if len(info['area_history']) > N_HISTORY:
            info['area_history'].pop(0)
        
        # Calculate smoothed area (e.g., simple moving average)
        avg_area = sum(info['area_history']) / len(info['area_history'])
        info['avg_area'] = avg_area
        
        # Calculate area change ratio from previous frame to current (if available)
        if len(info['area_history']) >= 2:
            prev_area = info['area_history'][-2]
            info['area_change_ratio'] = area / prev_area if prev_area > 0 else 1.0
        else:
            info['area_change_ratio'] = 1.0  # no previous frame, so no change
        
        # Approximate distance based on initial reference (inverse-square law)
        if info.get('initial_area') and info['initial_area'] > 0:
            initial_area = info['initial_area']
            initial_marker_dist = info['initial_marker_distance']
            # distance ≈ initial_marker_dist * sqrt(initial_area / current_area)
            approx_dist = initial_marker_dist * ( (initial_area / area) ** 0.5 )
        else:
            approx_dist = float('inf')  # just in case (should not happen if initial_area is set)
        info['approx_distance'] = approx_dist
        
        # Determine threat: True if estimated obstacle distance is less than marker distance
        info['is_threat'] = (approx_dist < marker_distance)
        
        # Record last seen timestamp (optional, here using frame index if available)
        info['last_seen'] = frame.index if hasattr(frame, 'index') else None
        
        # Prepare the obstacle data to return for this frame
        detected_obstacles.append({
            'id': obj_id,
            'center': (cx, cy),
            'label': label,
            'area': area,
            'approx_distance': approx_dist,
            'is_threat': info['is_threat']
        })
    
    # (Optional) Cleanup: remove obstacles not seen for a while to keep history small
    # e.g., if an obstacle's last_seen is too old, delete it from obstacle_history.
    
    return detected_obstacles









def check_temporal_consistency(position, area):
    """
    Check if the current obstacle position and area are consistent with the last 5 frames to avoid false positives
    """
    for past_frame in obstacle_history[-min(5, len(obstacle_history)):]:
        for past_obstacle in past_frame.values():
            if (abs(past_obstacle['position'][0] - position[0]) < 15 and                      # Be in approximately the same position (within 15 pixels in both x and y directions)
                abs(past_obstacle['position'][1] - position[1]) < 15 and                      
                abs(past_obstacle['area'] - area) / area < 0.15):                             # Have a similar area (difference < 15% of current area)
                return 1.0
    return 0.0
    

def update_temporal_history(obstacles):
    """
    Update the obstacle history for temporal consistency in obstacle detection
    """
    global obstacle_history

    # Format current obstacles for history tracking 
    try:
        current_frame = {pos: {
            'position': pos,
            'area': data['area'],
            'distance_cm': data['distance_cm'],
            'confidence_score': data['confidence_score'],
            'path_proximity': data['path_proximity']
            } for pos, data in obstacles.items()}
    except Exception as e:
        print(f"Error in formatting current frame: {e}")
        return
    
    # Add current frame to history
    obstacle_history.append(current_frame)

    # Keep only the last 10 frames
    if len(obstacle_history) > 10:
        obstacle_history.pop(0)        


def avoid_obstacle(obstacles):
    """
    Avoid obstacles using Vector Field Histogram (VFH) approach.
    Take detected obstacles from detect_obstacle() and generate optimal RC commands.
    All distances are standardized to centimeters for consistency.
    """

    global frame, last_marker_position, marker_distance

    if not obstacles:
        return False

    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Create a 360-degree histogram with 36 bins (10 degrees per bin)
    angle_bins = 36
    bin_size = 360 / angle_bins
    histogram = [float("inf")] * angle_bins

    # Fill histogram with obstacle distances at different angles 
    for pos, data in obstacles.items():
        x, y = pos
        dx, dy = x - center_x, center_y - y                                                   # y is inverted in image coordinates
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360                                # Ensure positive angles between 0 and 360
        bin_idx = int(angle // bin_size)

        # Update bin with minimum distance
        distance_cm = data['distance_cm']
        confidence = data['confidence_score']

        # Weight distance by confidence score (higher confidence = more important obstacle)
        effective_distance_cm = distance_cm / (0.5 + confidence * 0.5)                              

        # Update bin with minimum distance (in cm)
        histogram[bin_idx] = min(histogram[bin_idx], effective_distance_cm)
 
    # Apply smoothing to the histogram to avoid abrupt changes
    smoothed_histogram = smooth_histogram(histogram)

    # Find the best direction to move based on the smoothed histogram
    best_direction, is_path_clear = find_best_direction(smoothed_histogram, last_marker_position, center_x, center_y)

    # Convert directions to RC commands
    lr, fb, ud, yaw = direction_to_rc_commands(best_direction, is_path_clear)

    # Send combined RC commands
    tello.send_rc_control(lr, fb, ud, yaw)

    # Return True if still avoiding obstacles, False if path is clear
    return not is_path_clear

def smooth_histogram(histogram, window_size=3):
    """
    Apply smoothing to histogram to avoid abrupt changes in directions.
    """
    smoothed = histogram.copy()
    n = len(histogram)

    for i in range(n):
        # Calculate average of neighboring bins
        total = 0
        count = 0
        for j in range(-window_size // 2, window_size // 2 + 1):
            idx = (i + j) % n                                                                 # Wrap around for circular histogram
            if histogram[idx] != float("inf"):
                total += histogram[idx]
                count += 1
        if count > 0:
            smoothed[i] = total / count

    return smoothed


def find_best_direction(histogram, marker_position, center_x, center_y):
    """
    Find valleys in the obstacle histogram to determine the optimal direction to move.
    Returns:
        best_direction: The optimal direction to move (in degrees 0 to 359)
        is_path_clear: Boolean indicating if there is a clear path ahead
    """

    angle_bins = len(histogram)
    bin_size = 360 / angle_bins

    # Find all candidate directions (valleys in the histogram)
    valleys = []
    threshold_distance = 50                                                                   # Minimum distance to consider a direction safe

    for i in range(angle_bins):
        if histogram[i] > threshold_distance:
            # Find the extent of this valley
            start_idx = i
            while histogram[i] > threshold_distance:
                i = (i + 1) % angle_bins
                if i == start_idx:                                                            # Full circle, all directions are clear           
                    break

            end_idx = (i - 1) % angle_bins
            valley_width = (end_idx - start_idx) % angle_bins + 1
            valley_center = (start_idx + valley_width // 2) % angle_bins

            # Calculate average distance in this valley
            count = 0
            valley_distance = 0
            for j in range(valley_width):
                idx = (start_idx + j) % angle_bins
                if histogram[idx] != float("inf"):
                    valley_distance += histogram[idx]
                    count += 1
            
            if count > 0:
                valley_distance /= count
            
            valleys.append({
                'center': valley_center,
                'width': valley_width,
                'distance': valley_distance
            })

    # If no valleys found, all directions are blocked
    if not valleys:
        return 0, False                                                                       # Default direction, path is not clear

    # Calculate direction to marker
    marker_direction = None
    if marker_position is not None:
        dx, dy = marker_position[0] - center_x, center_y - marker_position[1]                 # y is inverted in image coordinates
        marker_direction = (math.degrees(math.atan2(dy, dx)) + 360) % 360                     # Ensure positive angles between 0 and 360
        marker_bin = int(marker_direction // bin_size)

    # Choose the best valley
    best_valley = None
    min_cost = float("inf")

    for valley in valleys:
        valley_direction = valley['center'] * bin_size

        # Cost function: balance between valley width, distance, and alignment with marker
        width_factor = min(1.0, valley['width'] / 5)                                          # Prefer wider valleys
        distance_factor = min(1.0, 200/valley['distance']) if valley['distance'] > 0 else 0   # Prefer longer distances

        # Direction cost: prefer valleys aligned with the marker
        direction_cost = 0
        if marker_direction is not None:
            angle_diff = min((valley_direction - marker_direction) % 360, 
                             (marker_direction - valley_direction) % 360)
            direction_cost = angle_diff / 180.0         
            
        # Combine cost (lower is better)                                      
        cost = (1 - width_factor) + (1 - distance_factor) + direction_cost

        if cost < min_cost:
            min_cost = cost
            best_valley = valley
    
    # Convert valley center to direction in degrees
    best_direction = best_valley['center'] * bin_size

    # Determine if path is clear 
    is_path_clear = best_valley['width'] >= 3 and best_valley['distance'] > 100 

    return best_direction, is_path_clear


def direction_to_rc_commands(direction, is_path_clear):
    """
    Convert direction to RC commands 

    Args:
        direction: The optimal direction to move (in degrees 0 to 359)
        is_path_clear: Boolean indicating if there is a clear path ahead
    Returns:
        lr, fb, ud, yaw: RC commands for left/right, forward/backward, up/down, and yaw
    """
    base_speed = 20

    # If path is clear, move forward in best direction
    if is_path_clear:
        # Convert direction to yaw and forward motion
        # 0° = right, 90° = forward, 180° = left, 270° = backward
        lr = int(base_speed * math.cos(math.radians(direction)))
        fb = int(base_speed * math.sin(math.radians(direction)))    
        ud = 0
        yaw = 0
    else:
        # If path is blocked, rotate in place to find a clear path
        yaw_direction = direction - 90                                                        # Adjust for drone's forward direction
        if yaw_direction < -180:
            yaw_direction += 360
        elif yaw_direction > 180:
            yaw_direction -= 360
        
        # Scale yaw based on how far we need to turn
        yaw_magnitude = min(1.0, abs(yaw_direction) / 45.0) * base_speed
        yaw = int(math.copysign(yaw_magnitude, yaw_direction))
                  
        # Move slightly backward if very close to obstacles
        lr = 0
        fb = -10  # Slight backward movement
        ud = 0

    return lr, fb, ud, yaw

                  
def search_with_memory(tello, last_marker_position, last_marker_direction, last_marker_time, timeout):
    """
    Search for ArUco marker using memory of its last known position and direction.
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
            tello.send_rc_control(lr_speed, fb_speed, ud_speed, yaw_speed)                    # Move in predicted direction
        else:
            print("No direction available.")
            tello.send_rc_control(0, 0, 0, 20)                                                # Default yaw rotation if no direction is available
    else:
        print("No memory available.")
        tello.send_rc_control(0, 0, 0, 20)                                                    # Default search pattern if no memory is available

# -----------------------------
# Initiate the state machine
# -----------------------------
STATE_TRACKING = "tracking"
STATE_AVOIDANCE = "avoidance"
STATE_SEARCHING = "searching"
mode = None

def capture_frames():
    """
    State machine to control the drone's behavior based on the current mode and detected obstacles by continuously capturing frames and listening for key presses.
    Press 's' to takeoff the drone.
    Press 'l' to land the drone.
    Press 't' to activate tracking mode.
    Press 'q' to quit the video stream.
    """
    global frame, quit_program, flying, mode
    try:
        while True:
            frame = tello.get_frame_read().frame
            # Ensure the frame is valid
            if frame is not None and frame.size > 0:         
                # Correct the frame for camera distortion
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                
                try:
                    drone_height = tello.get_height()                                         # Get current height in cm
                except Exception as e:
                    drone_height = -1
                
                height_text = f"Height: {drone_height} cm" if drone_height != -1 else "Height: N/A"  
                cv2.putText(frame, height_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Display height on the frame

                cv2.imshow("Drone Feed", frame)  
                
                # State machine behavior
                if mode == STATE_SEARCHING:
                    marker_found = track_aruco_marker()
                    if marker_found:
                        print("Marker found! Switching to TRACKING mode.")
                        mode = STATE_TRACKING
                    else:
                        print("Searching for marker...")
                        search_with_memory(tello, last_marker_position, last_marker_direction, last_marker_time, marker_memory_timeout)
                # switch to tracking once marker is found
                elif mode == STATE_TRACKING:
                    obstacles = detect_obstacle(frame)
                    if any(obstacles.values()):
                        print("Obstacle detected! Switching to AVOIDANCE mode.")
                        mode = STATE_AVOIDANCE
                    else:
                        marker_found = track_aruco_marker()
                        if not marker_found:
                            print("Marker lost! Switching to SEARCHING mode.")
                            mode = STATE_SEARCHING

                # swtich to avoidance once obstacle is detected
                elif mode == STATE_AVOIDANCE:
                    print("Executing avoidance maneuver...")
                    obstacles = detect_obstacle(frame)
                    still_avoiding = avoid_obstacle(obstacles)
                    if not still_avoiding:
                        marker_found = track_aruco_marker()
                        if marker_found:
                            print("Obstacle cleared! Marker found! Switching to TRACKING mode.")
                            mode = STATE_TRACKING
                        else:
                            print("Obstacle cleared! Marker lost! Switching to SEARCHING mode.")
                            mode = STATE_SEARCHING

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):     
                    cv2.destroyAllWindows()                                                   # Close all OpenCV windows
                    quit_program = True
                    break
                elif key == ord('s'):
                    if not flying:
                        tello.takeoff()                                     
                        tello.send_rc_control(0, 0, 40, 0)                                    # Move up 40 and typically reach at 100cm 
                        flying = True
                        print("Drone takeoff!")
                    else:
                        print("Drone is already airborne.")
                elif key == ord('l'):
                    if flying:
                        tello.land()
                        flying = False
                        print("Drone landing!")
                    else:
                        print("Drone is already landed.")
                elif key == ord('t'):
                    print("SEARCHING mode activated.")
                    mode = STATE_SEARCHING

            # short sleep can help with CPU usage and frame timing.
            time.sleep(0.1)  

    except cv2.error as e:
        print(f"OpenCV error: {e}")
    except Exception as e:
        print(f"General error: {e}")                                                         


# -----------------------------
# Main loop (No multithreading)
# -----------------------------
flying = False      # Keep track of whether the drone is airborne
quit_program = False

try:
    while not quit_program:
        # command = shared_command[0].lower()
        capture_frames()
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    if flying:
        try:
            tello.land()
            print("Drone landing!")
        except Exception as e:
            print("Error sending land command: ", e)
    try:
        tello.streamoff()
        tello.end()
    except Exception as e:
        print("Error ending Tello connection: ", e)




