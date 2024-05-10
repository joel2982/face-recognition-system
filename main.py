import face_recognition
import cv2
import numpy as np
from datetime import datetime
import csv

video_capture = cv2.VideoCapture(0)

# Elon Musk
elon_img = face_recognition.load_image_file('face-recognition-system\\faces\\ElonMusk.jpeg')
elon_encoding = face_recognition.face_encodings(elon_img)[0]
# Mark Zuckerberg
mark_img = face_recognition.load_image_file('face-recognition-system\\faces\\MarkZuckerberg.jpg')
mark_encoding = face_recognition.face_encodings(mark_img)[0]
# Sam Altman
sam_img = face_recognition.load_image_file('face-recognition-system\\faces\\SamAlman.jpg')
sam_encoding = face_recognition.face_encodings(sam_img)[0]
# Satya Nadella
satya_img = face_recognition.load_image_file('face-recognition-system\\faces\\SatyaNadella.jpg')
satya_encoding = face_recognition.face_encodings(satya_img)[0]
# Sundar Pichai
sundar_img = face_recognition.load_image_file('face-recognition-system\\faces\\SundarPichai.jpg')
sundar_encoding = face_recognition.face_encodings(sundar_img)[0]
# Tim Cook
tim_img = face_recognition.load_image_file('face-recognition-system\\faces\\TimCook.jpg')
tim_encoding = face_recognition.face_encodings(tim_img)[0]

student_face_encodings = list(elon_encoding ,mark_encoding, sam_encoding, satya_encoding, sundar_encoding, tim_encoding) 
student_names = list('Elon Musk', 'Mark Zuckerberg', 'Sam Altman', 'Satya Nadella', 'Sundar Pichai', 'Tim Cook')

students = student_names.copy()
face_locations = []
face_encodings = []

current_date = datetime.now().strftime(f'%Y-%m-%d')

file = open(f'{current_date}.csv','w+',newline='')
line_writer = csv.writer(file)

    