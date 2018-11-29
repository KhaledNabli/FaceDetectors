import numpy as np
import cv2


import face_recognition
import glob
from PIL import Image, ImageDraw
import numpy as np

cap = cv2.VideoCapture(1)


known_images = []
known_persons = []
known_faces = []


for filename in glob.glob('faces\\*.jpg'): #assuming gif
    person_name = filename.split("\\")[-1][:-4].title()
    im=face_recognition.load_image_file(filename)
    fe=face_recognition.face_encodings(im)[0]

    print("Init Face {0} for {1}".format(filename, person_name))

    known_images.append(im)
    known_persons.append(person_name)
    known_faces.append(fe)





while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)



    face_names = []
    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        for idx, val in enumerate(match):
            if val:
                name = known_persons[idx]

        face_names.append(name)



    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom ), (right, bottom+35), (255, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 3, bottom + 20), font, 0.8, (10, 10, 10), 1)


    # Display the resulting frame
    cv2.imshow('frame',frame)


    # quit when press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()