# from djitellopy import Tello
# import speech_recognition as sr
# import cv2
# import numpy as np
# import threading
# import time

# # Camera parameters
# Camera_fx = 916.4798394056434                                                                           # Focal lengths in pixels (how much the lens magnifies the image)
# Camera_fy = 919.9110948929223                                                                           
# Camera_cx = 483.57407010014435                                                                          # Principal point (the optical center of the camera, usually near the center of the image)
# Camera_cy = 370.87084181752994

# # Distortion coefficients
# Camera_k1 = 0.08883662811988326                                                                         # Radial distortion (causes straight lines to appear curved)  
# Camera_k2 = -1.2017058559646074                                                                         # Large negative distortion, likely causing "barrel distortion" 
# Camera_k3 = 4.487621066094839
# Camera_p1 = -0.0018395141258008667                                                                      # Tangential distortion (causes the image to look tilted or skewed)     
# Camera_p2 = 0.0015771769902803328 

# # Forming the camera matrix and distortion coefficients
# camera_matrix = np.array([[Camera_fx, 0, Camera_cx], [0, Camera_fy, Camera_cy], [0, 0, 1]], dtype="double")
# dist_coeffs = np.array([Camera_k1, Camera_k2, Camera_p1, Camera_p2, Camera_k3])

# tello = Tello()
# tello.connect()
# print(f"Battery: {tello.get_battery()}%")
# tello.streamoff()
# tello.streamon()

# # Global variables for voice commands and frame capture
# stop_voice_thread = False
# shared_command = [""]
# frame = None

# def listen_for_commands(shared_command):
#     global stop_voice_thread
#     r = sr.Recognizer()
#     while not stop_voice_thread:
#         with sr.Microphone() as source:
#             r.adjust_for_ambient_noise(source, duration=1)
#             print("Listening for voice commands...")
#             try:
#                 audio = r.listen(source, timeout=5, phrase_time_limit=5)  # Adjust timeout and phrase_time_limit as needed
#                 command = r.recognize_google(audio).lower()
#                 print(f"Recognized command: {command}")
#                 shared_command[0] = command
#             except sr.UnknownValueError:
#                 print("Sorry, I did not understand that.")
#             except sr.RequestError as e:
#                 print(f"Error recognizing the command: {e}")
#             except sr.WaitTimeoutError:
#                 print("Listening timeout, please speak again.")

# # Voice command listening thread
# voice_thread = threading.Thread(target=listen_for_commands, args=(shared_command,))
# voice_thread.start()

# def capture_frames():
#     global frame
#     while True:
#         frame = tello.get_frame_read().frame
#         if frame is not None:
#             frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
#             cv2.imshow("Tello Camera", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         time.sleep(1) 

# # Start the frame capture thread
# capture_thread = threading.Thread(target=capture_frames)
# capture_thread.start()

# try:
#     while True:
#         time.sleep(2)
# except KeyboardInterrupt:
#     print("Stopping...")
# finally:
#     stop_voice_thread = True
#     if voice_thread.is_alive():
#         voice_thread.join()
#     if capture_thread.is_alive():
#         capture_thread.join()
#     tello.streamoff()
#     cv2.destroyAllWindows() 

"""
Multithreading testing code FROM Chatgpt 
"""
import threading
import queue
import cv2
import speech_recognition as sr
from djitellopy import Tello

# Shared queue for communication (if needed in future)
command_queue = queue.Queue()

# Initialize drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()

def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        with mic as source:
            print("Listening for voice commands...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {command}")

                # Example: Send command to drone or UI
                command_queue.put(command)
                
                # Example response
                if "take off" in command:
                    print("Taking off...")
                    tello.takeoff()
                elif "land" in command:
                    print("Landing...")
                    tello.land()
                elif "exit" in command or "quit" in command:
                    print("Exiting voice recognition.")
                    break
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
              print(f"Speech recognition error: {e}")
            
def display_camera_feed():
    try:
        cv2.namedWindow("Tello Camera feed", cv2.WINDOW_NORMAL)     # Create a resizable window to display the camera feed
        while True:
            frame = tello.get_frame_read().frame 
            if frame is None or frame.shape[0] == 0: # Ensure frame is valid
                print("Error: No video frames received. Retrying...")
                continue
            frame = cv2.resize(frame, (640, 480))
            # Ensure OpenCV doesn't crash due to missing window manager
            cv2.waitKey(1)
            cv2.imshow("Tello Camera feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting video stream...")
                break
    except Exception as e:
        print(f"Error in video stream: {e}")
    finally:
        print("Shutting down stream...")
        tello.streamoff()
        cv2.destroyAllWindows()

voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
camera_thread = threading.Thread(target=display_camera_feed, daemon=True)

voice_thread.start()
print("Voice recognition thread started.")
camera_thread.start()
print("Camera feed thread started. Press 'q' to exit.")

# Keep main thread running
voice_thread.join()
camera_thread.join()

print("Program terminated.")