import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from PIL import Image
import random
import time

model = load_model(r"D:\A\shinkar\trimester4_project\Virtual_Cricket\Virtual_Cricket\face_detector\project.h5")

def winner(user_score,system_score):
    if user_score>system_score:
        return("User Won")
    elif user_score==system_score:
        return( "Tie")
    else:
        return("System Won")

def display_computer_move(system, frame):
    icon = cv2.imread( "sysscore/{}.png".format(system), 1)
    icon = cv2.resize(icon, (128,128))
    
    # This is the portion which we are going to replace with the icon image
    roi = frame[0:128, 0:128]

    # Get binary mask from the transparent image, 4th channel is the alpha channel 
    mask = icon[:,:,-1] 

    # Making the mask completely binary (black & white)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Store the normal bgr image
    icon_bgr = icon[:,:,:3] 
    
    # Now combine the foreground of the icon with background of ROI 
    
    img1_bg = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(mask))

    img2_fg = cv2.bitwise_and(icon_bgr, icon_bgr, mask = mask)

    combined = cv2.add(img1_bg, img2_fg)

    frame[0:128, 0:128] = combined

    return frame

# This list will be used to map probabilities to class names, Label names are in alphabatical order.
label_names = [0,1,2,3,4,5,6]
over=[]
attempts = 6
total_attempts=attempts
user_score=0
system_score=0
result=""
cap = cv2.VideoCapture(0)
prob_thrsh=0.7
flag=0
hand=False
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    #print(frame.shape)
   
    # Drawing the ROI
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    #roi = cv2.resize(roi, (768,768))
    roi=cv2.resize(roi,(128,128))
    # Normalize the image like we did in the preprocessing step, also convert float64 array.
    roi = np.array([roi]).astype('float64') / 255.0
    #roi = cv2.resize(roi,(128,128),Image.ANTIALIAS)
    test=[]
    test.append(roi)
    # Get model's prediction.
    pred = model.predict(test)
    
    # Get the index of the target class.
    target_index = np.argmax(pred[0])

    # Get the probability of the target class
    prob = np.max(pred[0])
    cv2.putText(frame, "prediction: {} {:.2f}%".format(label_names[np.argmax(pred[0])], prob*100 ),
                (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
    system=0
    user=0
    if prob>prob_thrsh and hand == False:
        #time.sleep(2)
        hand = True
        system=random.choice([1,2,3,4,5,6])
        user=label_names[np.argmax(pred[0])]
        if user in over:
            print("Show Different Number")
            cv2.putText(frame, "Show Different Number",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
            continue
        else:
            display_computer_move(system, frame)
            over.append(user)
            print("Adding new score",over)
        print(over)

        user_score=user_score+user
        system_score=system_score+system
        total_attempts -= 1
        #time.sleep(2)
        if system==user:
            print("Same score")
            flag=1
            cv2.putText(frame, "Same score, OUT",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
            break
    else:
        hand=False
    #cv2.putText(frame, "Your Score: {}  System score: {}".format(user, system),
     #           (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)    
    cv2.putText(frame, "Your total score: {} System total score: {}".format(user_score,system_score),
                (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Attempt left:{}".format(total_attempts),
                (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 150, 255), 2, cv2.LINE_AA)
    if total_attempts == 0:
        #result=winner(user_score,system_score)
        break
    cv2.imshow("Game", frame)
   
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
if flag==0:
    result=winner(user_score,system_score)
    print("System score: {} user Score: {}".format(system_score,user_score))
    print("Result:",result,"With",(abs(user_score-system_score)),"Runs")
else:
    print("******OUT********")
    print("System Won")
cap.release()
cv2.destroyAllWindows()