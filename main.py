import face_recognition
import cv2
import numpy as np
from datetime import datetime
import csv

video_capture = cv2.VideoCapture(0)

# Elon Musk
elon_img = face_recognition.load_image_file('faces\\ElonMusk.jpeg')
elon_encoding = face_recognition.face_encodings(elon_img)[0]
# Mark Zuckerberg
mark_img = face_recognition.load_image_file('faces\\MarkZuckerberg.jpg')
mark_encoding = face_recognition.face_encodings(mark_img)[0]
# Sam Altman
sam_img = face_recognition.load_image_file('faces\\SamAltman.jpg')
sam_encoding = face_recognition.face_encodings(sam_img)[0]
# Satya Nadella
satya_img = face_recognition.load_image_file('faces\\SatyaNadella.jpg')
satya_encoding = face_recognition.face_encodings(satya_img)[0]
# Sundar Pichai
sundar_img = face_recognition.load_image_file('faces\\SundarPichai.jpg')
sundar_encoding = face_recognition.face_encodings(sundar_img)[0]
# Tim Cook
tim_img = face_recognition.load_image_file('faces\\TimCook.jpg')
tim_encoding = face_recognition.face_encodings(tim_img)[0]

student_face_encodings = [elon_encoding ,mark_encoding, sam_encoding, satya_encoding, sundar_encoding, tim_encoding]
student_names = ['Elon Musk', 'Mark Zuckerberg', 'Sam Altman', 'Satya Nadella', 'Sundar Pichai', 'Tim Cook']

students = student_names.copy()
face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime(f'%Y-%m-%d')

file = open(f'{current_date}.csv','w+',newline='')
line_writer = csv.writer(file)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings=student_face_encodings, face_encoding_to_check=face_encoding)
        face_distance = face_recognition.face_distance(face_encodings=student_face_encodings, face_to_compare=face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = student_names[best_match_index]
            cv2.putText(img=frame, text=f'{name} Present', org=(10,100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=3, lineType=2)
            if name in students:
                students.remove(name)
                now = datetime.now()
                current_time = now.strftime("%H-%M-%S")
                line_writer.writerow([name, current_time])
                print(name)
                print(students)
            
    cv2.imshow('Attendance',frame)
    if cv2.waitKey(1) & 0xFF == ord('n'):
         break

video_capture.release()
cv2.destroyAllWindows()
file.close()
