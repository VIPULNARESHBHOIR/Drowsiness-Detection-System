import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance 
import pyttsx3 


engine=pyttsx3.init('sapi5')
voice=engine.getProperty('voices')
engine.setProperty('voices',voice[0].id)
engine.setProperty('rate',190)


def robo(text):
    engine.say(text)
    engine.runAndWait()


def eye_aspect_ratio(eye):
    A=distance.euclidean(eye[1],eye[5])
    B=distance.euclidean(eye[2],eye[4])
    C=distance.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear

thresh=0.25
flag=0
frame_check=40
(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detect=dlib.get_frontal_face_detector()
predict=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame=imutils.resize(frame,width=500)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    subjects=detect(gray,0)
    for sub in subjects:
        shape=predict(gray,sub)
        shape=face_utils.shape_to_np(shape)
        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEar=eye_aspect_ratio(leftEye)
        rightEar=eye_aspect_ratio(rightEye)
        ear=(leftEar+rightEar)/2.0
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
        if ear<thresh:
            flag=flag+1
            print(flag)
            if flag>=frame_check:
                cv2.putText(frame,"!!!!!ALERT!!!!!",(40,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
                cv2.putText(frame,"Need to take CUP OF TEA",(20,300),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,225),2)
                robo("sir, you are sleeping. wake up please. you need to take cup of tea.")
        else:
            flag=0
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cv2.destroyAllWindows()
cap.release()
