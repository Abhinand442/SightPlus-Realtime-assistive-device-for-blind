import cv2
import pytesseract
import pyttsx3

import book_detection

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

tts_engine = pyttsx3.init()

video_path = 0

cap = cv2.VideoCapture(video_path)


last_frame_book = False
similarity_threshold = 180
frame_skip_interval = 50  
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or cannot read the video file.")
        break

    if frame_counter % frame_skip_interval == 0:
        book_detected, objects_detected = book_detection.detect_book_in_frame(frame)
        if not(book_detected == True and last_frame_book == True):
            for object in objects_detected:
                tts_engine.say(object)
                tts_engine.runAndWait()

        if book_detected == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            text = pytesseract.image_to_string(gray).strip()
                
            tts_engine.say(text)
            tts_engine.runAndWait()
                
    frame_counter += 1
    last_frame_book = book_detected
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()