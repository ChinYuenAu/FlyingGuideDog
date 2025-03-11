"""
` cv, Aruco changed, Dictionary_get() --> getPredefinedDictionary()
` need to install ffmpeg and its dependency to decode incoming video stream.
` Consider replace speech_recognition with Vosk, gTTS with pyttsx3
` distance(cm) = (marker_width(cm) * focal_length(px)) / sqrt(marker_area) 
"""

from djitellopy import Tello
import speech_recognition as sr
import cv2
import numpy as np
import threading
import time
import requests
from gtts import gTTS
import pygame
import os
import json
import pyaudio

# Server endpoints and token
server_url = 'https://dev-api.youdescribe.org/upload'
server_url_chat = 'https://dev-api.youdescribe.org/chat'
token = 'VVcVcuNLTwBAaxsb2FRYTYsTnfgLdxKmdDDxMQLvh7rac959eb96BCmmCrAY7Hc3'

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
Camera_p2 = 0.0015771769902803328                                                                       # (caused by misalignment of the lens and image sensor)


# Forming the camera matrix and distortion coefficients
camera_matrix = np.array([[Camera_fx, 0, Camera_cx], [0, Camera_fy, Camera_cy], [0, 0, 1]], dtype="double")
dist_coeffs = np.array([Camera_k1, Camera_k2, Camera_p1, Camera_p2, Camera_k3])

# Initialize the Tello drone
tello = Tello()
tello.connect()
tello.for_back_velocity = 0
tello.left_right_velocity = 0
tello.up_down_velocity = 0
tello.yaw_velocity = 0
tello.speed = 0

print(f"Battery: {tello.get_battery()}%")
tello.streamoff()
tello.streamon()

time.sleep(5)  # Allow time for frames to accumulate

# Global variables for voice commands and frame capture
stop_voice_thread = False
shared_command = [""]
frame = None

def listen_for_commands(shared_command):
    global stop_voice_thread
    r = sr.Recognizer()
    while not stop_voice_thread:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            print("Listening for voice commands...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)  # Adjust timeout and phrase_time_limit as needed
                command = r.recognize_google(audio).lower()
                print(f"Recognized command: {command}")
                shared_command[0] = command
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError as e:
                print(f"Error recognizing the command: {e}")
            except sr.WaitTimeoutError:
                print("Listening timeout, please speak again.")

# Voice command listening thread
voice_thread = threading.Thread(target=listen_for_commands, args=(shared_command,))
voice_thread.start()

def track_aruco_marker():
    global pError, frame
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
        
        # Draw the center
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

        # Tracking logic here
        pError = trackObj(tello, (center_x, center_y), area, frame.shape[1], pid, pError)

def capture_frames():
    global frame
    try:
        while True:
            frame = tello.get_frame_read().frame
            print("frame: ", frame)
            # ensure the frame is valid
            if frame is not None and frame.size > 0:         
                # lock for safe access to frame 

                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                
                if tracking:
                    track_aruco_marker()

                # Display the frame
                print("frame: ", frame)
                # break
                cv2.imshow("Drone Feed", frame)  

                print("everything above line 137 looks fine")
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the video stream
                    break
            time.sleep(0.1)  # Adjust sleep time as needed for frame rate
            print("everything above line 154 looks fine")
    except cv2.error as e:
        print(f"OpenCV error: {e}")
    except Exception as e:
        print(f"General error: {e}")

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# ArUco marker tracking variables and parameters
frameWidth = 360  # or the actual width of your video frame

# PID parameters
pid = [0.1, 0, 0.01]                                                                          # Proportional→Reacts to current error, Integral→Accumulates past errors (set to 0 here), Derivative→Predicts future error trends
pError = 0                                                                                    # Previous error (difference from center), used for smooth PID-based tracking
fbRange = [4000, 6000]                                                                        # acceptable range for the detected object's area in pixels, too small→drone forward, too large→drone backward, within→hover
# Get the ArUco dictionary and parameters
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)                               
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)                        # retrieve predifined dictionary of 6x6 Aruco markers with 250 unique IDs 
aruco_params = cv2.aruco.DetectorParameters()                                                 # create a set of parameters for marker detection (Thresholds for binarization, Minimum/maximum marker sizes, Corner refinement settings)
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)                            # Need to instantiate ArucoDetector object from cv 4.7.x                                     
    
def trackObj(tello, marker_center, area, frame_width, pid, pError):
    fb = 0                                                                                    # forward-backward movement
    x, y = marker_center
    error = x - frame_width // 2                                                              

    yaw = pid[0] * error + pid[1] * (error - pError)                                          # Yaw rotation using PID control
    yaw = int(np.clip(yaw, -100, 100))                                                        # Limits yaw speed between -100 (full left) and 100 (full right) to prevent overcorrection

    if fbRange[0] <= area <= fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -10
    elif area < fbRange[0]:
        fb = 10
    
    if x == 0:
        yaw = 0
        error = 0

    tello.send_rc_control(0, fb, 0, yaw)                                                      # left_right_velocity, forward_backward_velocity, up_down_velocity, Rotate_left_right_velocity
    return error                                                                              # returns new error, which will be used as pError in the next frame for PID smoothing

def get_caption(image_path):
    with open(image_path, 'rb') as fileBuffer:                                                # Opens image file in binary mode (rb) to prepare it for uploading
        multipart_form_data = {
            'token': ('', token),
            'image': (os.path.basename(image_path), fileBuffer),                              # multipart form data: token + filename, file content
        }

        response = requests.post(server_url, files=multipart_form_data, timeout=10)
        if response.status_code == 200:
            json_obj = response.json()
            return json_obj['caption']
        else:
            print(f"Server returned status {response.status_code}")
            return "No caption received"

def text_to_speech(text, filename):
    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():                                                      # True when music is playing. Loop continues until playback finishes
        pygame.time.Clock().tick(10)                                                          # ensures the loop checks roughly 10 times per second, providing delay to avoid a busy-wait loop that hogs CPU

def capture_and_process_images():
    directions = ['front', 'right', 'back', 'left']
    global direction_image_paths
    direction_image_paths = {}
    output_directory = 'output'
    speech_files = []

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, direction in enumerate(directions):
        time.sleep(1)  # Delay between captures
        # Capture and save image
        if i == 0:
            time.sleep(3)
        image_name = os.path.join(output_directory, f'{direction}_{int(time.time())}.jpg')
        cv2.imwrite(image_name, frame)
        
        # Get the caption for the image
        caption = get_caption(image_name)
        full_caption = f"In {direction}: {caption}"
        print(f"Caption for {direction}: {full_caption}")

        # Convert caption to speech and save
        speech_filename = os.path.join(output_directory, f'caption_{direction}.mp3')
        text_to_speech(full_caption, speech_filename)

        direction_image_paths[direction] = image_name
        speech_files.append(speech_filename)
        
        # Rotate drone for each direction
        tello.rotate_clockwise(90)
        time.sleep(2)  # Adjust delay as needed

        #time.sleep(1)  # Delay between captures
    
    for speech_file in speech_files:
        play_audio(speech_file)

def send_question_and_get_response(question, image_path):
    data = {
        'token': token,
        'message': question,
        'image': (os.path.basename(image_path), open(image_path, 'rb')),
    }

    response = requests.post(server_url_chat, data=data)
    if response.status_code == 200:
        answer_output = json.loads(response.text)
        return answer_output.get('botReply', 'No answer received')
    else:
        print("Failed to send question to server:", response.status_code)
        return "No answer received"

"""
For offline testing (beginning)
"""
# print("Drone started successfully!")
# tello.takeoff()
# time.sleep(3)
# tello.move_up(50)

# while True: 
#     print("Enter 'Track' to start tracking or 'Stop' to quit:")
#     command = input()
    
#     if command == "Track":
#         tracking = True
#         print("Tracking mode activated.")
#         track_aruco_marker()

#     elif command == "Stop":
#         print("Stopping the drone...")
#         if capture_thread.is_alive():
#             capture_thread.join()
#         tello.streamoff()
#         tello.land()
#         cv2.destroyAllWindows()
#         break

"""
For offline testing (end)
"""

# Placeholder for the main loop
try:
    tracking = False
    checking_surroundings = False

    while True:
        command = shared_command[0].lower()
        
        if command == "take off":
            #Drone takeoff and move to head level
            print("Drone started successfully!")
            tello.takeoff()
            tello.streamon()
            # time.sleep(1)
            # tello.move_up(50) #Adjust height as needed

        elif command == "tracking":
            tracking = True
            checking_surroundings = False
            print("Tracking mode activated.")

        elif command == "check for surrounding":
            tracking = False
            checking_surroundings = True
            tello.move_up(20) #Adjust height as needed
            capture_and_process_images()
        
        elif command.startswith("in front") or command.startswith("to your right") or command.startswith("behind") or command.startswith("to your left"):
            # Extract the direction and question
            direction, _, question = command.partition(' ')
            image_path = direction_image_paths.get(direction)
            if image_path:
                response = send_question_and_get_response(question, image_path)
                text_to_speech(response, 'response.mp3')

        elif command == "landing":
            print("Stopping the drone...")
            stop_voice_thread = True
            voice_thread.join()
            tello.streamoff()
            tello.land()
            cv2.destroyAllWindows()
            break
        
        elif command == "quit":
            print("Quitting...")
            stop_voice_thread = True
            voice_thread.join()
            tello.streamoff()
            cv2.destroyAllWindows()
            break


        if tracking:
            print("Tracking mode activated.")
            track_aruco_marker()
            
        # Reset the command
        shared_command[0] = ""
        time.sleep(0.5)
        

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    stop_voice_thread = True
    if voice_thread.is_alive():
        voice_thread.join()
    if capture_thread.is_alive():
        capture_thread.join()
    tello.land()
    tello.streamoff()

