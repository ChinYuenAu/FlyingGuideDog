from djitellopy import Tello
import cv2
import numpy as np
import pygame
import os
import json
import time
import random

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
fbRange = [29680, 36650]                                                            
# Range of marker areas in pixels for forward/backward movement (tune this value based on your needs)
# fbRange = [4000, 6000] (Original)
# distance(cm) = (marker_width(cm) * focal_length(px)) / sqrt(marker_area) 
# (9.4 * 916.4798394056434) / sqrt(46380) = 40.00  || (9.4 * 916.4798394056434) / sqrt(36650) = 45.00  || (9.4 * 916.4798394056434) / sqrt(29680) = 50.00 
# (9.4 * 916.4798394056434) / sqrt(24530) = 55.00  || (9.4 * 916.4798394056434) / sqrt(20610) = 60.00

# ------------------------
# Global variable
# ------------------------
frame = None
# For marker memory
last_marker_direction = None
last_marker_position = None
last_marker_time = 0
marker_memory_timeout = 1.0  # Duration in seconds to remember marker direction and position


def track_aruco_marker():
    """
    Detect and track the ArUco marker in the video frame.
    """
    global pError_x, pError_y, frame, last_marker_position, last_marker_direction, last_marker_time
    marker_found = False
    if frame is None:
        return marker_found

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # corners: list of marker corners detected in the image
    # ids: unique IDs of the detected markers
    # rejected: markers detected but didn't meet certain criteria (like size, orientation, etc.)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)                           

    if ids is not None and len(corners) > 0:
        marker_found = True                                                                   # Marker detected
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)                                    # Draw the detected marker
        corner = corners[0][0]                                                                # Calculates the bounding box of the marker
        x, y, w, h = cv2.boundingRect(corner)                                                                                 
        area = w * h                                                                          # Calculate area and center
        center_x = int((corner[:, 0].sum()) / 4)
        center_y = int((corner[:, 1].sum()) / 4)
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)                           # Draw the center of the marker

        # Update the marker memory
        current_time = time.time()
        if last_marker_position is not None:
            # Calculate the direction of the marker based on the previous position
            dx = center_x - last_marker_position[0]
            dy = center_y - last_marker_position[1]
            last_marker_direction = (dx, dy)

        last_marker_position = (center_x, center_y)
        last_marker_time = current_time

        # Tracking logic here
        pError_x, pError_y = trackObj(tello, (center_x, center_y), area, frame.shape[0], frame.shape[1], pid_x, pid_y, pError_x, pError_y)

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
    Detect obstacles in the drone's path using Canny edge detection and contour analysis by dividing the frame into regions with proximity information.
    """
    # Implement obstacle detection logic here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200) 

    # Divide frame into regions (top, bottom, left, right, center)
    height, width = edges.shape
    regions = {
        "top": edges[0: height//3, :],
        "middle": edges[height//3: 2*height//3, :],
        "bottom": edges[2*height//3:, :],
        "left": edges[:, 0:width//3],
        "center": edges[:, width//3: 2*width//3],
        "right": edges[:, 2*width//3:],
    }

    # Check each region for obstacles
    obstacles = {}
    for region_name, region_img in regions.items():
        contours, _ = cv2.findContours(region_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacle_detected = False
        max_proximity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:
                obstacle_detected = True
                # calculate proximity as ratio of contour area to region area
                region_area = region_img.shape[0] * region_img.shape[1]
                proximity = min(1.0, area / (region_area * 0.5))                                # Scale factor to keep proximity between 0 and 1
                max_proximity = max(max_proximity, proximity)

        obstacles[region_name] = (obstacle_detected, max_proximity)
    
    return obstacles


def avoid_obstacle(obstacles):
    """
    Avoid obstacles by sending RC commands with speed adjustments based on proximity information from different regions.
    """
    # Default speed values
    lr, fb, ud, yaw = 0, 0, 0, 0

    # Base speed for different maneuvers
    base_speed = {
        "lr": 20,
        "fb": 20,
        "ud": 0,
    }

    # Dynamic obstacle avoidance based on detected regions
    if obstacles["top"][0] and not obstacles["bottom"][0]:                                    # Obstacle in the top region, move down
        proximity = obstacles["top"][1]
        ud = -int(base_speed["ud"] * (0.5 + proximity * 0.5))                                 # Scale from 50% to 100% of base speed
    elif obstacles["bottom"][0] and not obstacles["top"][0]:                                  # Obstacle in the bottom region, move up
        proximity = obstacles["bottom"][1]
        ud = int(base_speed["ud"]) * (0.5 + proximity * 0.5)                                  
    if obstacles["left"][0] and not obstacles["right"][0]:                                    # Obstacle in the left region, move right               
        proximity = obstacles["left"][1]
        lr = int(base_speed["lr"] * (0.5 + proximity * 0.5))
    elif obstacles["right"][0] and not obstacles["left"][0]:                                  # Obstacle in the right region, move left
        proximity = obstacles["right"][1]
        lr = -int(base_speed["lr"] * (0.5 + proximity * 0.5))
    
    # If obstacle in center or multiple directions blocked
    if obstacles["center"][0]:
        proximity = obstacles["center"][1]
        fb = -int(base_speed["fb"] * (0.5 + proximity * 0.5))
    elif (obstacles["top"][0] and obstacles["bottom"][0]) or (obstacles["left"][0] and obstacles["right"][0]):
        # Calculate average proximity for complex situations
        avg_proximity = 0.0
        count = 0
        for region in ["top", "bottom", "left", "right"]:
            if obstacles[region][0]:
                avg_proximity += obstacles[region][1]
                count += 1
        if count > 0:
            avg_proximity /= count
            fb = -int(base_speed["fb"] * (0.5 + avg_proximity * 0.5))
    
    # Send the combined RC commands
    tello.send_rc_control(lr, fb, ud, yaw)

    # Return True if the drone is still avoiding obstacles, False if the path is clear.
    return any(obstacles[region][0] for region in obstacles)


def search_with_memory(tello, last_marker_position, last_marker_direction, last_marker_time, timeout):
    """
    Search for ArUco marker using memory of its last known position and direction.
    """
    current_time = time.time()
    if last_marker_position is not None and (current_time - last_marker_time) < marker_memory_timeout:
        if last_marker_direction is not None:
            dx, dy = last_marker_direction
            yaw_speed = int(np.clip(dx * 0.2, -30, 30))                                             # Scale x-directino to yaw speed
            ud_speed = int(np.clip(dy * 0.1, -30, 30))                                              # Scale y-direction to up/down speed   
            lr_speed = int(np.clip(dx * 0.1, -30, 30))                                              # Scale x-direction to left/right speed (drifting)
            fb_speed = 10
            tello.send_rc_control(lr_speed, fb_speed, ud_speed, yaw_speed)                           # Move in predicted direction
        else:
            tello.send_rc_control(0, 0, 0, 20)                                                      # Default yaw rotation if no direction is available
    else:
        tello.send_rc_control(0, 0, 0, 20)                                                          # Default search pattern if no memory is available

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
                # swtich to tracking once marker is found
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
                        tello.takeoff()                                                       # typically at 80cm 
                        tello.move_up(40)
                        flying = True
                        print("Drone takeoff!")
                    else:
                        print("Drone is already airborne.")
                elif key == ord('l'):
                    tello.land()
                    print("Drone landing!")
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
