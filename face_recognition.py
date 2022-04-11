#We will use KNN algo for this face detection project
#1.) load the training data (numpy array file of every person's data stored)
#     x -> values stored in numpy array
#     y -> value assigned to that array(each person)
#2.) Read a video using openCV
#3.) Extract the faces out of it(Test faces)
#4.)use KNN algo to find the predicted face
#5.)map the predicted id to the name of the person(make a dictionary)
#6.)Display the prediction on the screen with images with their respective names


import cv2
import numpy as np
import os

########### KNN algo ##############
#finding the Euclidian distance
def distanceFun(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

def knnAlgo(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]

        d = distanceFun(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key = lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]

    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]
#####################################


#Init camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './face_data/'
face_data_nparr = []
label = []

class_id = 0  #labels for the given file
name = {}     #mapping between id and name


#Data Prepration
#os.listdir() shows all the files in the given location of folder
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #Creating mapping btw class_id and name
        name[class_id] = fx[:-4]
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data_nparr.append(data_item)

        #creating labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data_nparr, axis=0)
face_labels = np.concatenate(label, axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

#Testing

while True:
    ret,frame = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x,y,w,h = face

        #Get the region of intrest
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        #Predicted label
        op = knnAlgo(trainset, face_section.flatten())

        #Display the prediction on the screen with images with their respective names
        pred_name = name[int(op)]
        cv2.putText(frame, pred_name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255),2)

    cv2.imshow("Your Face",frame)

    key_pressed = cv2.waitKey(1) & 0xff
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

