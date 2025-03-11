from djitellopy import Tello
import cv2
import numpy as np
import pygame
import os
import json
import time

# Camera parameters
Camera_fx = 916.4798394056434                                                                           # Focal lengths in pixels (how much the lens magnifies the image)
Camera_fy = 919.9110948929223                                                                           
Camera_cx = 483.57407010014435                                                                          # Principal point (the optical center of the camera, usually near the center of the image)
Camera_cy = 370.87084181752994

# Distortion coefficients
Camera_k1 = 0.08883662811988326                                                                         # Radial distortion (causes straight lines to appear curved)  
Camera_k2 = -1.2017058559646074                                                                         # Large negative distortion, likely causing "barrel distortion" 
Camera_k3 = 4.487621066094839
Camera_p1 = -0.0018395141258008667                                                                      # Tangential distortion (causes the image to look tilted or skewed)     
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

# PID and tracking parameters
pid_x = [0.2, 0, 0.02]                                                                          # Proportional gain→Reacts to current error, Integral→Accumulates past errors (set to 0 here), Derivative gain→Increases if drone overshoots or oscillates too much.
pid_y = [0.1, 0, 0.02]                                                                          # PID gains for vertical control
# Previous errors: used to calculate the derivative term for smooth PID-based tracking
pError_x = 0                                                                                    # For yaw control (horizontal movement)
pError_y = 0                                                                                    # For vertical control (up/down movement) 
pError_area = 0                                                                                 # For forward/backward control (distance from marker)
prev_fb = 0
# fbRange = [4000, 6000]  
fbRange = [29680, 36650]  
# distance(cm) = (marker_width(cm) * focal_length(px)) / sqrt(marker_area) 
# (9.4 * 916.4798394056434) / sqrt(46380) = 40.00  || (9.4 * 916.4798394056434) / sqrt(36650) = 45.00  || (9.4 * 916.4798394056434) / sqrt(29680) = 50.00 
# (9.4 * 916.4798394056434) / sqrt(24530) = 55.00  || (9.4 * 916.4798394056434) / sqrt(20610) = 60.00

# Global variable
frame = None
tracking = False

def track_aruco_marker():
    global pError_x, pError_y, frame
    if frame is None:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # corners: list of marker corners detected in the image, ids: unique IDs of the detected markers, rejected: markers detected but didn't meet certain criteria (like size, orientation, etc.)
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    # corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)                               

    if ids is not None and len(corners) > 0:
        # Draw the detected marker
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Calculates the bounding box of the marker
        corner = corners[0][0]
        x, y, w, h = cv2.boundingRect(corner)                                                                                 

        # Calculate area and center
        area = w * h
        center_x = int((corner[:, 0].sum()) / 4)
        center_y = int((corner[:, 1].sum()) / 4)
        
        # Draw the center point of the marker
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

        # Tracking logic here
        pError_x, pError_y = trackObj(tello, (center_x, center_y), area, frame.shape[0], frame.shape[1], pid_x, pid_y, pError_x, pError_y)

def capture_frames():
    """
    Continuously capture frames from the Tello drone and display them.
    Press 's' to takeoff the drone.
    Press 'l' to land the drone.
    Press 't' to activate tracking mode.
    Press 'q' to quit the video stream.
    """
    global frame, quit_program, tracking, flying
    try:
        while True:
            frame = tello.get_frame_read().frame
            # ensure the frame is valid
            if frame is not None and frame.size > 0:         
                # correct the frame for camera distortion
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                cv2.imshow("Drone Feed", frame)  

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):     
                    cv2.destroyAllWindows()                                                   # Close all OpenCV windows
                    quit_program = True
                    break
                elif key == ord('s'):
                    if not flying:
                        tello.takeoff()
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
                    print("Tracking mode activated.")
                    tracking = True

            if tracking:
                print("Tracking mode activated.")
                track_aruco_marker()
                    
            # a short sleep can help with CPU usage and frame timing.
            time.sleep(0.1)  

    except cv2.error as e:
        print(f"OpenCV error: {e}")
    except Exception as e:
        print(f"General error: {e}")                                                          # acceptable range for the detected object's area in pixels, too small→drone forward, too large→drone backward, within→hover

def trackObj(tello, marker_center, area, frame_height, frame_width, pid_x, pid_y, pError_x, pError_y, dt=0.1):
    """
    Control both yaw (rotational) and up/down movement using PD controllers

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
    yaw = int(np.clip(yaw, -80, 80))                                                        # Limits yaw speed between -100 (full left) and 100 (full right) to prevent overcorrection

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
    # tello.send_rc_control(0, fb, ud, yaw)           

    # return the current error so it can be used in the next cycle as pError                                         
    return error_x, error_y                                                

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
