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

# Server endpoints and token
server_url = 'https://dev-api.youdescribe.org/upload'
server_url_chat = 'https://dev-api.youdescribe.org/chat'
token = 'VVcVcuNLTwBAaxsb2FRYTYsTnfgLdxKmdDDxMQLvh7rac959eb96BCmmCrAY7Hc3'

# Camera parameters
Camera_fx = 916.4798394056434
Camera_fy = 919.9110948929223
Camera_cx = 483.57407010014435
Camera_cy = 370.87084181752994

# Distortion coefficients
Camera_k1 = 0.08883662811988326
Camera_k2 = -1.2017058559646074
Camera_p1 = -0.0018395141258008667
Camera_p2 = 0.0015771769902803328
Camera_k3 = 4.487621066094839

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
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None and len(corners) > 0:
        # Draw the detected marker
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

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
    while True:
        frame = tello.get_frame_read().frame
        if frame is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            if tracking:
                track_aruco_marker()

            cv2.imshow("Drone Feed", frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the video stream
                break
        time.sleep(0.1)  # Adjust sleep time as needed for frame rate

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# ArUco marker tracking variables and parameters
frameWidth = 360  # or the actual width of your video frame

# PID parameters
pid = [0.1, 0, 0.01]
pError = 0
fbRange = [4000, 6000]
# Get the ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()
    
def trackObj(tello, marker_center, area, frame_width, pid, pError):
    fb = 0
    x, y = marker_center
    error = x - frame_width // 2

    yaw = pid[0] * error + pid[1] * (error - pError)
    yaw = int(np.clip(yaw, -100, 100))

    if fbRange[0] <= area <= fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -10
    elif area < fbRange[0]:
        fb = 10
    
    if x == 0:
        yaw = 0
        error = 0

    tello.send_rc_control(0, fb, 0, yaw)
    return error

def get_caption(image_path):
    with open(image_path, 'rb') as fileBuffer:
        multipart_form_data = {
            'token': ('', token),
            'image': (os.path.basename(image_path), fileBuffer),
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
    while pygame.mixer.music.get_busy(): 
        pygame.time.Clock().tick(10)

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

# Placeholder for the main loop
try:
    tracking = False
    checking_surroundings = False

    while True:
        command = shared_command[0].lower()
        
        if command == "start":
            #Drone takeoff and move to head level
            print("Drone started successfully!")
            tello.takeoff()
            time.sleep(3)
            tello.move_up(50) #Adjust height as needed

        elif command == "start tracking":
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

        elif command == "stop":
            print("Stopping the drone...")
            stop_voice_thread = True
            voice_thread.join()
            tello.streamoff()
            tello.land()
            cv2.destroyAllWindows()
            break

        if tracking:
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
