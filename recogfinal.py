import numpy as np
import cv2
from tensorflow  import keras
from tensorflow .keras.preprocessing.image import ImageDataGenerator
#import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
import numpy as np


detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'emotion.h5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'emotion.h5'
model = keras.models.load_model("./best_model.h5")
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 50
ROI_left = 250

word_dict = {0:'Zero',1:'one',2:'Two',3:'Super',4:'Victory',5:'How are you?',6:'A',7:'B',8:'C',9:'Hungry',10:'Hey',11:'Loser',12:'Good'}
pred=0
cnt=0
count=0
laststate=0
def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)



def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

cam = cv2.VideoCapture(0)
amount_of_frames = cam.get(cv2.CAP_PROP_FPS)
print(amount_of_frames)
num_frames =0
while True:
    if(count!=15):
        ret, frame = cam.read()
        count=count+1
        continue
    else:
        count=0
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
    
        frame_copy = frame.copy()
    
        # ROI from the frame
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    
    
        if num_frames < 70:
            
            cal_accum_avg(gray_frame, accumulated_weight)
            
            cv2.putText(frame_copy, "TAKING BACKGROUND", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        else: 

            frame = imutils.resize(frame,width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            
            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frameClone = frame.copy()
            if len(faces) > 0:
                faces = sorted(faces, reverse=True,
                key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                    # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48,48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
            else: continue
        
         
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                        # construct the label text
                        text = "{}: {:.2f}%".format(emotion, prob * 100)                      
                        w = int(prob * 300)
                        cv2.rectangle(canvas, (7, (i * 35) + 5),
                        (w, (i * 35) + 35), (0, 0, 255), -1)
                        cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
                        cv2.putText(frame_copy, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                        cv2.rectangle(frame_copy, (fX, fY), (fX + fW, fY + fH),
                                      (0, 0, 255), 2)
        
        


            # segmenting the hand region
            hand = segment_hand(gray_frame)
            
    
            # Checking if we are able to detect the hand...
            if hand is not None:
                
                thresholded, hand_segment = hand
    
                # Drawing contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
                
                cv2.imshow("Thesholded Hand Image", thresholded)
                
                thresholded = cv2.resize(thresholded, (64,64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
                
                pred = model.predict(thresholded)
                cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
        # Draw ROI on frame_copy
        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    
        # incrementing the number of frames for tracking
        num_frames += 1
    
        # Display the frame with segmented hand
        cv2.imshow("Sign Detection", frame_copy)
    
    
        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF
    
        if k == 27:
            break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
