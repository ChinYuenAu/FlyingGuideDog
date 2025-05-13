import time
import cv2
import numpy as np
import torch
import json
from djitellopy import Tello
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ----- Camera parameters -----
Camera_fx = 916.4798394056434                                                                 # Focal lengths in pixels (how much the lens magnifies the image)
Camera_fy = 919.9110948929223                                                                           
Camera_cx = 483.57407010014435                                                                # Principal point (the optical center of the camera, usually near the center of the image)
Camera_cy = 370.87084181752994
camera_matrix = np.array([[Camera_fx, 0, Camera_cx],
                          [0, Camera_fy, Camera_cy], 
                          [0, 0, 1]], dtype="double")

# ----- Distortion coefficients -----
Camera_k1 = 0.08883662811988326                                                               # Radial distortion (causes straight lines to appear curved)  
Camera_k2 = -1.2017058559646074                                                               # Large negative distortion, likely causing "barrel distortion" 
Camera_k3 = 4.487621066094839
Camera_p1 = -0.0018395141258008667                                                            # Tangential distortion (causes the image to look tilted or skewed)     
Camera_p2 = 0.0015771769902803328    
dist_coeffs = np.array([Camera_k1, Camera_k2, Camera_p1, Camera_p2, Camera_k3])

# ----- Load YOLO, Deep SORT Tracker and MiDaS -----
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
scripted_model = torch.jit.trace(model.model, torch.zeros(1, 3, 640, 640))                    # Creates a TorchScript version of YOLOv5n model for faster inference, trace the model with a dummy input tensor
scripted_model.save('yolov5n_scripted.pt')
scripted_model = torch.jit.load('yolov5n_scripted.pt').eval()                                 # Load the scripted model and set it to evaluation mode to disable training-specific behaviors like dropout and batch normalization
CLASS_NAMES = model.names if hasattr(model, 'names') else model.model.names                   # Get class names from the model
with open("yolov5n_classnames.json", "w") as f:
    json.dump(CLASS_NAMES, f)                                                                 # Save class names to a JSON file
with open("yolov5n_classnames.json", "r") as f:     
    CLASS_NAMES = json.load(f)                                                                # Load class names from the JSON file
tracker = DeepSort(max_age=20, n_init=3, max_cosine_distance=0.2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).to(device).eval()   # Load MiDaS_small model via PyTorch Hub to device and set to eval mode
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform                   # Preprocessing pipeline for MiDaS_small model
prev_obs_centers = {}                                                                         # Dictionary to store previous obstacle centers for filtering fast-moving objects

# ----- Detfine L/R depth ROI bounds for MiDaS -----
def get_roi_bounds(w, h):
    """
    Returns two tuples for left and right ROIs.
    """
    global marker_distance

    if marker_distance < 120.0:                                                               # ROIs are narrower when close to the marker to avoid false positives
        fraction_x = 0.1                                                                      # From 0 to 0.1 and 0.9 to 1 of frame width for L/R ROIs
    else:                                                                           
        fraction_x = 0.2                                                                      # From 0.2 to 0.8 of frame width for L/R ROIs

    fraction_y = 0.3                                                                          # From 0.3 to 0.7 of frame height for both ROIs
    left_roi = (0, int(h * fraction_y),                                                       
                 int(w * fraction_x), int(h * (1 - fraction_y)))      
    right_roi = (int(w * (1 - fraction_x)), int(h * fraction_y),                              
                 w, int(h * (1 - fraction_y))) 

    return left_roi, right_roi

# ----- Initialize Tello drone -----
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamoff()
tello.streamon()
frame_reader = tello.get_frame_read()                                                         # Get the frame reader object
flying = False                                                                                # Keep track of whether the drone is airborne
quit_program = False

while (frame_reader.frame is None) or (frame_reader.frame.size == 0):                         # Wait until the first frame is available
    time.sleep(0.1)                                              
print("First frame received.")                            

# ----- ArUco marker tracking setup -----
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)                        # Retrieve predifined dictionary of 6x6 Aruco markers with 250 unique IDs 
aruco_params = cv2.aruco.DetectorParameters()                                                 # Create a set of parameters for marker detection (Thresholds for binarization, Minimum/maximum marker sizes, Corner refinement settings)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)  

# ----- PID and tracking parameters -----
# Proportional gain: Reacts to current error
# Integral gain: Accumulates past errors (set to 0 here) 
# Derivative gain: Increases if drone overshoots or oscillates too much.
pid_x = [0.25, 0, 0.02]                                                                       # PID gains for yaw control was 0.2, 0, 0.02 (original)
pid_y = [0.2, 0, 0.02]                                                                        # PID gains for vertical control
# Previous errors: used to calculate the derivative term for smooth PID-based tracking
pError_x, pError_y = 0, 0                                                                     # For yaw control (horizontal movement) and vertical control (up/down movement)  
prev_fb = 0                                                                                   # Previous forward/backward speed    
prev_yaw = 0                                                                                  # Previous yaw speed
prev_lr = 0                                                                                   # Previous left/right speed                    
pError_dist = 0                                                                               # forward/backward (distance) previous error
marker_width_cm = 9.4                                                                         # Adjust to your marker's actual width(cm)
BASE_TARGET_DISTANCE_CM = 120                                                                 # Target distance from the camera to the marker in cm 
TURN_TARGET_DISTANCE_CM = 60                                                                  # Reduce radius of the turn

# For marker memory
last_marker_direction = None
last_marker_position = None
last_marker_time = 0
marker_memory_timeout = 5.0                                                                   # Duration in seconds to remember marker direction and position in search mode
last_avoid_roll = 0                                                                           # Last roll command for obstacle avoidance 
frame = None                                                                                  # Current video frame from the drone's camera
marker_distance = 150.0                                                                       # Default distance to the marker in cm                                 

# For YOLO + Deep SORT fallback
yolo_fallback_track_id = None
yolo_fallback_center = None


def track_aruco_marker(frame):
    """
    Track ArUco marker in video feed and update tracking info.

    Parameters:
        frame : Current video frame from the drone's camera

    Returns:
        marker_found : Boolean indicating if the marker was found
        (lr, fb, ud, yaw) : Control commands for left/right, forward/backward, up/down, and yaw
    """

    global pError_x, pError_y, last_marker_position, last_marker_direction, last_marker_time, marker_distance

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
            
            # Columns of R are the marker's local axes projected onto the camera's frame
            # R = [r_x, r_y, r_z] (each r_i is a 3x1 vector in the camera's coordinate system)
            # R[:. 0] is the x-axis of the marker in the camera's frame
            # R[:, 1] is the y-axis of the marker in the camera's frame
            # R[:, 2] is the z-axis of the marker in the camera's frame
            R, _ = cv2.Rodrigues(rvec)                                                        # Convert rotation vector to rotation matrix
            yaw_marker = np.arctan2(R[1, 0], R[0, 0])                                         # Extract yaw angle from rotation matrix, yaw around Z-axis in a Tait-Bryan ZYX rotation sequence                                          
            yaw_marker = yaw_marker - np.pi / 2                                               # Adjust yaw to be relative to the drone's frame of reference
            cv2.putText(frame, f"Yaw: {np.degrees(yaw_marker):.1f} deg",                      # Positive yaw is clockwise, negative is counter-clockwise. Degrees ≈ radians × 57.3
                        (int(corner[0][0]), int(corner[0][1]) - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            print(f"[DEBUG] dist={marker_distance:5.1f} cm")
        else:
            yaw_marker = 0.0                                                                  # Safe fallback if position estimation fails

        # Bounding box & center calculation
        x, y, w, h = cv2.boundingRect(corner)                        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)                          # Draw the bounding box of the marker                                                         
        area = w * h                                                                          # Calculate area and center
        
        # Compute center of marker   
        center_x = int((corner[:, 0].sum()) / 4)
        center_y = int((corner[:, 1].sum()) / 4)

        # Update previous position and direction for marker tracking
        current_time = time.time()
        if last_marker_position is not None:
            dx = center_x - last_marker_position[0]
            dy = center_y - last_marker_position[1]
            last_marker_direction = (dx, dy)                                                  # Calculate direction of marker based on previous position 
        last_marker_position = (center_x, center_y)
        last_marker_time = current_time

        # Tracking logic here
        pError_x, pError_y, lr, fb, ud, yaw = trackObj((center_x, center_y),
                                                        marker_distance, 
                                                        frame, 
                                                        pid_x, pid_y, pError_x, pError_y, yaw_marker)

    # Reset Yolo fallback if marker is found
    if marker_found:                                                                              
        global yolo_fallback_track_id, yolo_fallback_center
        yolo_fallback_track_id = None                                                             
        yolo_fallback_center = None

    return marker_found, (lr, fb, ud, yaw)                                                        # Return marker found status and tracking info


def trackObj(marker_center, marker_distance, frame, pid_x, pid_y, pError_x, pError_y, yaw_marker, dt=0.1):
    """
    PD Controller for marker centering and distance control.

    Parameters:
        marker_center   : Tuple (x, y) representing the marker center in pixels
        marker_distance : Distance from the camera to the marker in cm
        frame           : Current video frame from the drone's camera
        pid_x           : List of PID gains [Kp, Kd] for yaw rotation (Integral not used here)
        pid_y           : List of PID gains [Kp, Kd] for vertical control
        pError_x        : Previous horizontal error (for yaw derivative term)
        pError_y        : Previous vertical error (for up/down derivative term)
        yaw_marker      : Yaw angle of the marker in radians
        dt              : Time interval between frames/control updates

    Returns:
        new_pError_x, new_pError_y : Updated previous errors for the next iteration
        lr, fb, ud, yaw            : Control commands for left/right, forward/backward, up/down, and yaw
    """ 
    global prev_fb, pError_dist, prev_yaw, prev_lr
    x, y = marker_center
    frame_height, frame_width, _ = frame.shape                                                # Get the current frame's dimensions
    
    # ----- Horizontal (Yaw) control -----
    error_x = x - (frame_width // 2)                                                          # Error between the center of the marker and the center of the frame
    dError_x = (error_x - pError_x) / dt                                                      # derivative of horizontal error over time            
    yaw_pixel = pid_x[0] * error_x + pid_x[2] * dError_x                                      # PD control for yaw rotation    

    # ----- Vertical (Up/Down) control -----
    # Use the frame's height (global frame variable) for vertical error
    error_y = (frame_height // 2) - y                                                         # Positive error if marker is above the center, negative if below; In OpenCV, the y-axis increases downwards
    dError_y = (error_y - pError_y) / dt                                                      # derivative of vertical error over time
    ud = pid_y[0] * error_y + pid_y[2] * dError_y                                             # PD control for up/down movement

    # ----- Marker rotation control -----
    yaw_deg = np.degrees(yaw_marker)  # Convert once for reuse
    if abs(yaw_deg) < 10:
        yaw_marker = 0.0  # Treat as non-turning
        yaw_deg = 0.0
    # Define thresholds for pre-turn and full turn (e.g., hallway cornering)
    pre_turning = 10 <= abs(yaw_deg) < 20    # Early signs of actual turn
    turning     = abs(yaw_deg) >= 20        # Confident the user is turning (e.g., doorway exit or hallway corner)
    print(f"[DEBUG] yaw_marker={yaw_deg:.1f}°, turning={turning}, pre_turning={pre_turning}")

    # ----- Forward/Backward control -----   
    # Reduce the target distance if the marker is turning
    TARGET_DISTANCE_CM = (
        0.5 * BASE_TARGET_DISTANCE_CM + 0.5 * (TURN_TARGET_DISTANCE_CM if turning or pre_turning else BASE_TARGET_DISTANCE_CM)
    )
    print(f"[DEBUG] TARGET_DISTANCE_CM={TARGET_DISTANCE_CM:.1f} cm")

    err_d = marker_distance - TARGET_DISTANCE_CM                                              # positive error if farther than target distance, negative if closer
    dErr_d = (err_d - pError_dist) / dt                                                       # derivative of distance error over time
    kp_d = 0.3 if turning else 0.25                                          # Proportional gain for fb (increase this value to increase the speed of convergence to the target distance)                        
    kd_d = 0.06 if turning else 0.04                                                          # Derivative gain for fb (increase this value to reduce oscillations by reacting to the rate of change of the error)
    fb_dynamic = kp_d * err_d + kd_d * dErr_d                                                 # Dynamic forward/backward speed based on distance error
    pError_dist = err_d                                                                       # Update previous distance error for the next iteration
    
    # ----- Forward/Backward Deadzone & minimum speed -----
    min_speed = 10  
    fb = 0 if abs(err_d) < 5 else fb_dynamic if abs(fb_dynamic) >= min_speed else np.sign(fb_dynamic) * min_speed  

    # ----- Clip early, to prevent explosive swing ----- 
    if abs(err_d) > 40:
        yaw_pixel *= 0.5
    yaw_pixel = np.clip(yaw_pixel, -40, 40)                                                   

    # ----- Adaptive Scaling and Damping -----
    if turning:                                                                               # Aggressive damping if turning, cautious if far away from target distance, normal otherwise      
        yaw_marker_gain = 14
        lr_marker_gain = 26
        damping_factor = 1.2
    elif pre_turning:           # warmup zone
        yaw_marker_gain = 10     
        lr_marker_gain = 22
        damping_factor = 1.15
    else:
        yaw_marker_gain = 4
        lr_marker_gain = 10
        if abs(err_d) > 30:
            damping_factor = 0.4
        else:
            damping_factor = 1.0            

    curve_scale = min(max(marker_distance / 130.0, 1.0), 1.8)                                 # Wider arc if drone farther away from user
    
    pixel_weight = 0.2 if (turning or pre_turning) else (0.3 if abs(err_d) > 40 else 1.0)                                         # Reduce pixel influence and emphasize marker influence if significant marker rotation 
    marker_weight = 1.8 if (turning or pre_turning) else 1.0

    # Combine pixel-based control and marker-based control
    yaw_marker_corr = marker_weight * damping_factor * yaw_marker_gain * yaw_marker * curve_scale
    lr_marker_corr = marker_weight * damping_factor * lr_marker_gain * -yaw_marker * curve_scale     
    lr_pixel = 0.18 * error_x + 0.03 * dError_x                                           # was 0.12, 0.02 
    if abs(error_x) < 20:
        lr_pixel = 0
    elif abs(err_d) < 30:
        lr_pixel *= 0.4

    # Reduce lr_marker_corr influence when yaw_marker is tiny
    if not (turning or pre_turning):
        if abs(yaw_marker) < np.radians(4):
            lr_marker_corr = 0
        else:
            lr_marker_corr *= 0.3  # damp it further

    # Blend pixel-based and marker-based control
    if turning or pre_turning:                                                              
        lr = lr_marker_corr                                                                   # Aggressive marker yaw-based turn
    elif pre_turning:
        lr = 0.3 * lr_pixel + 0.7 * lr_marker_corr
    else:
        lr = 0.6 * lr_pixel + 0.4 * lr_marker_corr                                            # walking straight/stand still, was 0.4, 0.6
 
    if not turning and not pre_turning:                                                       # if not turning or pre-turning, use pixel-based control only
        yaw = yaw_pixel                                                                       
    else:
        # Combine marker yaw correction and pixel offset correction
        yaw = yaw_pixel * pixel_weight + yaw_marker_corr
                                                                    
    # ----- Low-pass filter to smooth the command -----
    alpha_lr = 0.3 if abs(err_d) > 30 else 0.6                                                # Smoothing factor (adjust as needed between 0 and 1); lower value give more smoothing
    alpha_fb = 0.6
    alpha_yaw = 0.6

    if turning or pre_turning:
        alpha_lr = 0.1
        alpha_fb = 0.2
        alpha_yaw = 0.2

    fb = alpha_fb * fb + (1 - alpha_fb) * prev_fb                            
    yaw = alpha_yaw * yaw + (1 - alpha_yaw) * prev_yaw
    lr  = alpha_lr * lr  + (1 - alpha_lr) * prev_lr
    prev_fb, prev_yaw, prev_lr = fb, yaw, lr

    # ----- Stability gate for vertical control -----
    if abs(err_d) < 10 and abs(error_y) < 30:
        ud = 0

    # ----- Turning override from steady state -----
    if abs(err_d) < 15:
        if turning or pre_turning:
            # Responsive override: react fast when turning
            lr = lr_marker_corr                                                               # Only marker yaw-based control for lr; pixel offset ignored
            yaw = yaw_pixel * pixel_weight + yaw_marker_corr
        else:
            # Hold position, suppress jitter
            lr *= 0.4
            yaw *= 0.4
            fb = 0
    elif abs(err_d) < 20:
        lr  *= 0.7
        yaw *= 0.7

    # ---- Anti-oscillation logic ----
    if not (turning or pre_turning) and (np.sign(error_x) != np.sign(pError_x)) and abs(err_d) < 10:
        yaw = 0
        lr = 0
    if not (turning or pre_turning) and (np.sign(error_y) != np.sign(pError_y)) and abs(err_d) < 5:
        ud = 0

    # ---- Dynamic clipping ----
    yaw = int(np.clip(yaw, -20, 20)) if abs(err_d) < 10 else int(np.clip(yaw, -45, 45))       # Clip yaw to a smaller range if close to the target distance
    lr  = int(np.clip(lr, -30, 30)) if abs(err_d) < 10 else int(np.clip(lr, -35, 35))         # Clip lr to a smaller range if close to the target distance
    fb = np.clip(fb, -60 + 0.2 * abs(error_x), 60 - 0.3 * abs(error_x))                       # Prevent forward thrust from overpowering correction
    ud = int(np.clip(ud, -30, 30))                                                            # Clip the speed to the Tello's limits (typically -100 to 100)     

    print(f"[DEBUG] error_x={error_x}, error_y={error_y}, err_d = {err_d:.2f}, lr={lr}, fb={fb}, ud={ud}, yaw={yaw}")  # Print debug info
    print(f"[DEBUG] yaw_marker={yaw_marker:.3f} rad, yaw_marker_corr={yaw_marker_corr:.2f}")
    print(f"[DEBUG] gains: yaw_gain={yaw_marker_gain}, lr_gain={lr_marker_gain}, pixel_weight={pixel_weight:.2f}")

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

    start_total = time.time()                                                                 # Measure total processing time per frame 

    # Run YOLO detection and prepare Deep SORT input
    # start_yolo = time.time()                                                                 
    # results = model(frame)[0]
    # end_yolo = time.time()                                 

    # Resize and preprocess frame
    img = cv2.resize(frame, (640, 640))                                                       # Resize the frame to 640x640 for YOLOv5n default input size
    img = img.transpose(2, 0, 1) / 255.0                                                      # Transpose from HWC to CHW (Channel, Height, Width) format expected by PyTorch and normalize pixel values to [0, 1] required by YOLOv5n
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)                                 # Convert NumPy array to PyTorch tensor. Unsqueeze(0) add batch dimension. YOLO expects batched output. Shape is [batch_size, channels, height, width]
    img = img.to(device)                                                                      # Move the tensor to the device (CPU/MPS/GPU) 

    start_yolo = time.time()
    with torch.no_grad():
        pred = scripted_model(img)[0]
    end_yolo = time.time()

    print("YOLO device:", next(model.parameters()).device)                                    # It will show "mps:0" if it is MPS, and "cpu" if it is CPU.
    print("YOLOv5n TorchScript inference time: {:.2f} ms".format((end_yolo - start_yolo) * 1000))
    print("YOLO GFLOPS: {:.2f}".format(model.info()['flops'] / 1e9))                          # Floating Point Operations per second, gives an estimate of 
                                                                                              # how complex the model is in terms of computation, not how fast it runs. 
    detections = []
    classes = []

    for det in pred:
        x1, y1, x2, y2, conf, cls = map(float, det[:6])
        if conf > 0.5:
            detections.append([x1, y1, x2, y2, conf])
            class_name = CLASS_NAMES.get(int(cls), str(int(cls)))                             # Get class name from class ID, or use ID as string if not found
            classes.append(class_name)

    # Convert detections from [x1, y1, x2, y2, conf] to [[x, y, w, h], conf] to match Deep SORT format
    detections_converted = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        w = x2 - x1
        h = y2 - y1
        detections_converted.append([[x1, y1, w, h], conf])

    # Update Deep SORT tracker 
    tracks = tracker.update_tracks(detections_converted, frame=frame)

    track_classes = {}                                                                        # track_id -> class_name
    for track, cls_name in zip(tracks, classes):
        track_classes[track.track_id] = cls_name


    # -------------------------------
    # Run MiDaS Depth Estimation for lateral obstacle avoidance
    # -------------------------------
    start_midas = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                        # Convert frame from BGR->RGB
    input_batch = transform(frame_rgb).to(device)                                             # Apply MiDaS transforms
    with torch.no_grad():                                                                     # Disable gradient calculation for speedup, as now in inference mode, not training
        depth_output = midas(input_batch)                                                     # Pass preprocessed image to MiDaS model and get depth output tensor
        # Upsampling to original frame size
        depth_map = torch.nn.functional.interpolate(                                          # Resize shape from [batch, height, width] to [batch, 1, height, width] required by interpolation
            depth_output.unsqueeze(1),                                                        # Shape becomes [1, 1, H, W] for interpolation to work
            size=(h_frame, w_frame),                                                          # Size=(height, width) of the original frame
            mode="biliniear",                                                                 # Change from "bicubic" to "bilinear" as "bicubic" is not supported by MPS
            align_corners=False                                                               
        ).squeeze().cpu().numpy()                                                             # Removes singleton dimensions (shape becomes [H, W]) and moves tensor back to CPU as numpy array  
                                                                                              # for further processing in OpenCV              

    end_midas = time.time()
    print("MiDaS device: {input_batch.device}")                                               # It will show "mps:0" if it is MPS, and "cpu" if it is CPU.
    print("MiDaS inference time: {:.2f} ms".format((end_midas - start_midas) * 1000))

    end_total = time.time()
    print(f"Total frame latency: {(end_total - start_total)*1000:.1f} ms")

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
    cv2.putText(depth_disp, f"R Depth 50th: {depth_metric_right:.2f}", (x1R - 100, y1R - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.rectangle(frame, (x1L, y1L), (x2L, y2L), (0, 0, 255), 2)
    cv2.rectangle(frame, (x1R, y1R), (x2R, y2R), (0, 0, 255), 2)
    cv2.putText(frame, f"L Depth 50th: {depth_metric_left:.2f}", (x1L + 5, y1L - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
    cv2.putText(frame, f"R Depth 50th: {depth_metric_right:.2f}", (x1R - 100, y1R - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Height, battery and speed info
    drone_height = tello.get_height()  
    height_text = f"Height: {drone_height} cm" if drone_height != -1 else "Height: N/A" 
    cv2.putText(frame, height_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    battery_text = f"Battery: {tello.get_battery()}%" if tello.get_battery() != -1 else "Battery: N/A"
    cv2.putText(frame, battery_text, (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) 
    speed_text = f"X-Asis Speed: {tello.get_state_field('vgx')} cm/s" if tello.get_state_field('vgx') != -1 else "Speed: N/A"
    cv2.putText(frame, speed_text, (10, h_frame - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # -------------------------------
    # Compute YOLO-based lateral avoidance (for dynamic obstacles)
    # -------------------------------
    left_boundary   = int(w_frame * 0.2)
    right_boundary  = int(w_frame * 0.8)
    # top_boundary    = int(h_frame * 0.1)
    # bottom_boundary = int(h_frame * 0.9)
    left_error = right_error = 0                                                              # how much an obstacle on the left/right intrudes: measured by (x2 - left_boundary) or (right_boundary - x1)

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

    return int(np.clip(avoid_roll, -30, 30)), tracks, track_classes



def fallback_yolo_track_person(frame, tracks, track_classes, last_marker_position):
    """
    Select the closest visible person track to last Aruco marker position and compute RC control to follow it.
    Params:
        frame                : Current video frame from the drone's camera
        tracks               : List of detected tracks from Deep SORT
        track_classes        : Dictionary mapping track IDs to class names
        last_marker_position : The last known position of the marker
    Returns:
        rc : Control commands for left/right, forward/backward, up/down, and yaw
    """
    global pError_x, pError_y
 
    if last_marker_position is None:                                                          # No marker found yet, do nothing
        return None

    min_dist = float('inf')
    selected_center = None
    marker_x, marker_y = last_marker_position
    frame_h, frame_w = frame.shape[:2]

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_name = track_classes.get(track_id, None)
        if class_name != 'person':
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dist = np.hypot(cx - marker_x, cy - marker_y)                                         # Euclidean distance to the marker

        if dist < min_dist:
            min_dist = dist
            selected_center = (cx, cy)

    if selected_center is None:
        return None                                                                           # No person found to track
    
    # Apply PD control (same logic as in trackObj)
    x, y = selected_center
    error_x = x - (frame_w // 2)
    dError_x = (error_x - pError_x) / 0.1
    yaw = pid_x[0] * error_x + pid_x[2] * dError_x

     # Left/Right
    k_lr = 0.25
    lr = k_lr * error_x                                            

    # Forward (constant for now)
    fb = 5

    # Up/Down
    error_y = (frame_h // 2) - y
    dError_y = (error_y - pError_y) / 0.1
    ud = pid_y[0] * error_y + pid_y[2] * dError_y

    # Update previous errors
    pError_x, pError_y = error_x, error_y

    # Draw bounding box and track ID for visualization
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(frame, f'ID {track_id}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=2)

    yaw = int(np.clip(yaw, -15, 15))                                                          # Limits yaw speed between -50 (full left) and 50 (full right) to prevent overcorrection
    lr = int(np.clip(lr, -30, 30))       
    ud = int(np.clip(ud, -30, 30))

    # Return RC command
    return (lr, fb, ud, yaw)


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
# Main loop
# -----------------------------
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
                avoid_roll, current_tracks, track_classes = detect_obstacle(frame, scripted_model, tracker, midas, transform, device, prev_obs_centers)

                # Merge logic: prioritize obstacle avoidance over marker tracking
                if avoid_roll != 0:
                    # Obstacle avoidance has the highest priority
                    print("Obstacle detected! Avoiding...")
                    fb_avoid = max(10, int(0.5 * fb_marker))                                 # Adjust forward speed based on marker tracking
                    rc = (int(avoid_roll), int(fb_avoid), 0, int(yaw))                       # Roll command for obstacle avoidance
                elif marker_found:
                    # Use ArUco marker when available
                    print("ArUco Marker found! Tracking...")
                    rc = (int(lr), int(fb_marker), int(ud), int(yaw))
                else:
                    # If marker not found, fallback to YOLO + Deep SORT tracking
                    print("Marker not found! Attempting YOLO fallback...")
                    rc_fallback = fallback_yolo_track_person(frame, current_tracks, track_classes, last_marker_position)
                    if rc_fallback is not None:                
                        # If a person is found, use its position
                        rc = rc_fallback
                        print("Fallback tracking using YOLO + Deep SORT.")
                    else:
                        # If no marker and no person found, search using last known marker position and direction 
                        print("No marker found and no fallback available. Searching...")
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
                try:
                    tello.takeoff()
                    # for i in range(3):                                                            # Wait for Tello to stabilize
                    #     print(f"wait for 3 seconds for Tello to stabilize, {i+1} ...")
                    #     time.sleep(1)
                    height = tello.get_height()
                    print(f"Drone height after takeoff: {height}")
                except Exception as e:
                    print(f"[ERROR] Takeoff failed: {e}")

                tello.send_rc_control(0, 0, 30, 0)                                            # Ascend for 20 cm     
                for i in range(3):                                                            # Wait for Tello to stabilize
                        print(f"wait for 3 seconds for Tello to stabilize, {i+1} ...")
                        time.sleep(1)
                height = tello.get_height()
                print(f"Drone height: {height}")                                                    
                flying = True
                print("Flying flag:", flying)
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