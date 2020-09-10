import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from PIL import Image
import random
import time

model = load_model(r"D:\A\shinkar\trimester4_project\Virtual_Cricket\Virtual_Cricket\face_detector\project.h5")

def winner(user_score, system_score): return f'Result: User Won with {abs(user_score-system_score)} Runs' if user_score > system_score else f'Result: Tie' if user_score==system_score else f'Result: System Won with {abs(user_score-system_score)} Runs'

class Score():
    def __init__(self, user_score=0, system_score=0, over=[], system_array=[]):
        self.user_score = user_score
        self.system_score = system_array
        self.over = over
        self.system_array = system_array

    def __str__(self): return f'Result: User Won with {abs(self.user_score-self.system_score)} Runs' if self.user_score > self.system_score else f'Result: Tie' if self.user_score==self.system_score else f'Result: System Won with {abs(self.user_score-self.system_score)} Runs'

score = Score()

# This list will be used to map probabilities to class names, Label names are in alphabatical order.
label_names = [0,1,2,3,4,5,6]
over = []
attempts = 6
total_attempts=attempts
# parameter 1
user_score = 0 # user runs
# parameter 2
system_score = 0 # bot_runs
# parameter 3
result = "" # message result
system_array = []
cap = cv2.VideoCapture(0)
prob_thrsh=0.7
flag=0
try:
    while True:
        _, frame = cap.read()
        # mirror image
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
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
        
        if prob>prob_thrsh:
            time.sleep(1)
            system=random.choice([1,2,3,4,5,6])
            system_array.append(system)
            user=label_names[np.argmax(pred[0])]
            if user in over:
                print("Show Different Number")
                cv2.putText(frame, "Show Different Number",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
                continue
            else:
                over.append(user)
                print("Adding new score",over)
            print(over)

            user_score=user_score+user
            system_score=system_score+system
            total_attempts -= 1
            time.sleep(1)
            if system==user:
                print("Same score")
                flag=1
                cv2.putText(frame, "Same score, OUT",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
                break
        cv2.putText(frame, "Your Score: {}  System score: {}".format(user_score, system_score),
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)    
        cv2.putText(frame, "Your total score: {} System total score: {}".format(user_score,system_score),
                    (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Attempt left:{}".format(total_attempts),
                    (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
        if total_attempts == 0:
            # result=winner(user_score,system_score)
            break
        cv2.imshow("Game", frame)
        score.user_score, score.system_score, score.over, score.system_array = user_score, system_score, over, system_array
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    if flag==0:
        print(f"System score: {system_score}. User Score: {user_score}")
        result = winner(user_score, system_score)
        print(result)
        print(f'User Array: {over} System Array: {system_array}')
    else:
        print("******OUT********")
        print("System Won")
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()

# user_runs, bot_runs, message, user_runs_array, bot_runs_array
score.user_score, score.system_score, score.over, score.system_array = user_score, system_score, over, system_array