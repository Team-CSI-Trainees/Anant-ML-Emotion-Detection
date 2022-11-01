#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import sleep
from flask import Flask,render_template,Response
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras_preprocessing import image
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
app=Flask(__name__)
cap=cv2.VideoCapture(0)
face_classifier=cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
emotion_model = load_model('Resources/model.h5')

gender_model = load_model('Resources/gender_model_50epochs.h5')

class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
gender_labels = ['Male', 'Female']



def gen_frames():  
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            result = DeepFace.analyze(frame, actions=["age"])
            print(result)
            labels=[]
            detector = MTCNN()
            faces = detector.detect_faces(frame) 
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        #Get image ready for prediction
                roi=roi_gray.astype('float')/255.0  #Scale  
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)

                preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 7 classes
                label=class_labels[preds.argmax()]  #Find the label
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Gender
                roi_color=frame[y:y+h,x:x+w]
                roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
                gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
                gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
                gender_label=gender_labels[gender_predict[0]] 
                gender_label_position=(x,y+h+50) #50 pixels below to move the label outside the face
                cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Age
        
                cv2.putText(frame, str(result['age']),(x-25, y-25) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2 , cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
  app.run(debug=True)
    