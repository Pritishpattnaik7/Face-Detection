import cv2
import numpy as np

#Init camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data_nparr = []
dataset_path = './face_data/'
file_name = input("Enter the name of person : ")

while True:
    ret,frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key = lambda f:f[2]*f[3], reverse=True)  #sorting faces acc to the area(w*h)

    #drawing bounding rectangle
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

        #Extract (crop out the required face) : Region of intrest
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        skip += 1
        #after every 10th frame we are appending the image in the np array
        if skip%10==0:
            face_data_nparr.append(face_section)
            print(len(face_data_nparr))

    cv2.imshow("Video frame", frame)
    cv2.imshow("Face Section", face_section)



    key_pressed = cv2.waitKey(1) & 0xff
    if key_pressed == ord('q'):
        break

#convert our face list to a np array
face_data_nparr = np.asarray(face_data_nparr)
face_data_nparr = face_data_nparr.reshape((face_data_nparr.shape[0], -1))   #converting it to a 2d matrix
print(face_data_nparr.shape)

#saving the person's face to the face data folder as np file
np.save(dataset_path + file_name + '.npy', face_data_nparr)
print("Data Successfully saved at " + dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()