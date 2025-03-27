from djitellopy import Tello
import time
import speech_recognition as sr
import threading
import cv2
import numpy as np

"""
Testing video streaming
"""
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
# Camera_p2 = 0.0015771769902803328                                                                       # (caused by misalignment of the lens and image sensor)

# # Forming the camera matrix and distortion coefficients
# camera_matrix = np.array([[Camera_fx, 0, Camera_cx], [0, Camera_fy, Camera_cy], [0, 0, 1]], dtype="double")
# dist_coeffs = np.array([Camera_k1, Camera_k2, Camera_p1, Camera_p2, Camera_k3])

tello = Tello()

try:
    tello.connect()
    print(f"Battery: {tello.get_battery()}%") 
    tello.streamoff()
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(2)  # Allow time for frames to accumulate
    
    while True:
        frame = frame_reader.frame  # Get the latest frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame is None:
            print("Error: No video frames received. Retrying...")
            continue
        
        # Ensure frame is in uint8 format before undistortion
        # frame = frame.astype(np.uint8)

        print("Frame type: ", type(frame))
        print(frame)
        
        # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        cv2.imshow("Tello Stream", frame)
        # cv2.imshow("Tello Stream", undistorted_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("Program interrupted by user")


finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    tello.streamoff()
    tello.end()


"""
Testing speech recognition
"""
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





# import socket

# TELLO_IP = "192.168.10.1"
# TELLO_PORT = 8889
# TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.settimeout(5)

# try:
#     sock.sendto(b"command", TELLO_ADDRESS)
#     print("Sent: command")

#     response, _ = sock.recvfrom(1024)
#     print("Response:", response.decode())

# except socket.timeout:
#     print("No response from Tello. Check UDP connection.")

# finally:
#     sock.close()



import cv2
print(cv2.getBuildInformation())
