#!/usr/bin/env python
# coding: utf-8

# Importing and Installing dependencies

# In[3]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp 


# Keypoints using mp_holistic and mp_drawing

# In[4]:


# used for bringing the holistic model through .holistic          
mp_holistic = mp.solutions.holistic # Holistic model
# used for drawing the utilities - points and structure from the midiapipe which is its main function through .drawing_utils
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# In[5]:


# creating a function so that I do not have to write the above cell again and agian
# passing two variables image and model . image from the user and the model for mediapipe utilization

def mediapipe_detection(image, model):
    # so opencv reads the image in form of bgr but for detection using mediapipe we require the format to be RGB
    # so cv2.cvtcolor helps in recolouring the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable this helps us in saving a bit of memory
    # so here image is going to be a frame from video
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    # so that opencv can produce results in bgr format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# In[6]:


def draw_landmarks(image, results):
    # so we are passing it into the .draw_landmarks resulting the producing a structure to the image in the video
    # passing the image and results with respect to the lists of the various types of landmarks like face, left hand or right hand etc
    # this will provide us with the items present in the lists with a comprehensive details for the perticular landmarks section
    # mp_holistic is allowing us to pass the image via the connection map for a perticular landmark 
    # draw_landmark func does not return the image but rather applies the landmark visualizations to the current image in place.
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


# In[7]:


def draw_styled_landmarks(image, results):
    # Draw face connections
    # comes with the mediapipe a helper function mp_drawing Draws the landmarks and the connections on the image.
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), # color the joint 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) #color the connection
                             ) 
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), # color the joint 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) #color the connection
    #                          ) 
    
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


# In[8]:


mp_holistic.POSE_CONNECTIONS


# In[9]:


cap = cv2.VideoCapture(0)
# Set mediapipe model 
# so we are accesing the mediapipe model using the with mp_holistic.Holistic 
# so how the mediapipe model works is that it actuallly makes an initial detection using the min_detection_confidence 
# then track the key points with min_tracking_confidence=0.5 we can change it as well
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        # for entering the function
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # # Draw landmarks
        # helps in accessing the draw_landmarks func allowing to draw landmarks through mediapipe 
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[10]:


cap.release()
cv2.destroyAllWindows()


# Extracting keypoint values

# In[11]:


# results.
# here we have acquired the results parameters from the model like the mediapipe is effectively working
# as for eg:
results.pose_landmarks.landmark[0].visibility
# which helps in confirming the mediapipe model
# further here we are also conforming the "visibility" parameter of the landmark from pose through the test array formed  


# In[12]:


len(results.pose_landmarks.landmark)
# getting 33 because mediapipe offers 33 types of landmarks for the pose_landmarks func
# like nose , ears, shoulders wrists,etc


# In[13]:


# pose=[]
# for res in results.pose_landmarks.landmark:
#     test=np.array([res.x,res.y,res.z,res.visibility])
#     pose.append(test)


# so here we are basically forming an array pose which basically contains the parameters for the landmarks in the res variable obtained from the results from the pose_landmark
# again pose_landmarks denotes the func from the mediapipe lib and "landmark" shows the perticular landmark value out of those 33 parameters
# so we used .flatten to make it compatible for the LSTM model used further
pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)


# In[14]:


pose.shape
# conforming the shape of the array i.e 1 d


# In[15]:


# similarly
# here we donot have the visibility parameter for the hand so ignoring it 
lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
# small addition:-
# ERROR HANDLING
# now if the left hand is not present in the frame for that we need to handle the the case. for that we can form the array of 0's 

# similarly
rh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)


# In[16]:


# similarly for face
face_all_parameters=len(results.face_landmarks.landmark)*3
print(face_all_parameters)
face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)


# In[17]:


def extract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose,lh,rh,face])
# concatenating for the model to detect the sign language


# In[18]:


extract_keypoints(results).shape


# Setting up folders for collection

# In[19]:


# Path for exported data, numpy arrays
DATA_PATH=os.path.join('data for different actions')


# so what here we are going to do is that here the data will be collected and for that
# 30 number of sequences are taken into consideration which means that 30 videos worth of data
# for each sequence as well here 30 frames in length are taken into consideration which means 30*30 data
# again here we have for ex 3 gestures so the data becomes 30*30*3
# again here we have 1662 keypoints for the landmarks obtained earlier as a result the final data = 30*30*3*1662
# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_of_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


# In[20]:


# just creating the folders and sub folders
# action and seq in nested loop for forming folders

for action in actions: 
    for sequence in range(no_of_sequences):
        try: 
# makedirs used for making the sub directories
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


# Collecting keypoint values for Training nd Testing

# In[ ]:


# Set mediapipe model 
cap = cv2.VideoCapture(0)
# 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # NEW LOOP
    # Loop through actions
    # this loop is for specifications of the number of times the frames need to be saved for every sequence w.r.t every action
    for action in actions:
        # Looping through sequences aka videos
        for sequence in range(no_of_sequences):
            # Looping through video length aka sequence length (mentioned above) - can be changed
            for frame_num in range(sequence_length):

                # Reading the frame from the video. This frame will be used for further analysis
                ret, frame = cap.read() 
                # Making the detections using the mediapipe_detection func where the model will be able to process the frame like BGR to RGB 
                image, results = mediapipe_detection(frame, holistic)
                # print(results)
                # Drawing landmarks on the acquired frame
                draw_styled_landmarks(image, results)
                


                # logic is for the formating portion
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    # providing the break for adjusting the posture
                    cv2.waitKey(2000) #2 sec
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                

                # NEW Export keypoints
                # now for this frame or results obtained from mediapipe_extraction which will be RGB the keypoints will be extracted from the extract_keypoint func in the form of a 1 d array.
                # again in ideal senario the no of extracted points will be 1662 as found
                keypoints = extract_keypoints(results)
                # providing the path for the save
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # saving the array in the location
                np.save(npy_path, keypoints)

                # Break for this frame and continue for next respective iteration
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    # when loop ends the window closes
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


cap.release()
cv2.destroyAllWindows()


# Preprocessing data and creating labels w.r.t. actions

# In[21]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[22]:


# creating a dict where label is mapped with a num by def starting from 0
# enumerate calling the next label from the actions list , earlier created as the array
label_map = {label:num for num, label in enumerate(actions)}


# In[23]:


label_map


# In[24]:


# forming two arrays named as mentioned
# just as a graph where seq will contain all the data with respect to all the videos and frames recorded during the training
# and labels for denoting the actions

# now again this is our main task 
sequences, labels = [], []

# 3 actions so 3 iterations
for action in actions:
    # 30 videos with respect to each action so 30 iterations 
    for sequence in range(no_of_sequences):
        # forming a blank array for storing that x data of all the collection done till now
        window = []
        # for each frame recorded in each of the seq
        for frame_num in range(sequence_length):
            # so res is basically helping in loading the data for each frame through the os lib
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))) # frame_num shows the exact name to counter in the loop
            # appending the window array with res
            window.append(res)
        # now the loop for the frames for a perticular seq is over
        # adding the window data in the sequences array as a 2 d array with one parameter as each frame and second one as keypoints -1662
        sequences.append(window)

        # appending the labels array only once with only action i.e the action running the loop
        # values are going to be 1d because action is just a label which is currently 0 or 1 or 2 running action * seq times , here 30 
        # not any data containing 2 d array
        labels.append(label_map[action])


# In[25]:


np.array(sequences).shape


# In[26]:


X = np.array(sequences)


# In[27]:


X.shape


# In[28]:


# changing the labels from 0,1,2 to categorical data for easier accessebility
y = to_categorical(labels).astype(int)
y


# In[29]:


# so spliting the data into train and test with 5 percent of testing 
# data contains the seq with frames and keypoints respectively in form of a 3 d array   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
X_test.shape


# Building and training LSTM neural network

# In[30]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[31]:


# adding the logs folder
log_dir = os.path.join('Logs')
# tensorboard is a part of tensorflow monitoring the model training using a web app
# will help to track the accuracy during the training
tb_callback = TensorBoard(log_dir=log_dir)


# In[32]:


# neural network

# adding sequential API cuz it will allow in building the model fluidly
model = Sequential()
# adding the three layers of LSTM consisting of 3 positional argument and 1 keyword argument
# positional arg - depends on the position of the value. wrong position wrong output
# keyword arg - depends w.r.t the value assigned with the variable
# returning sequence is necessery because here if not then the next lstm layer will not follow the prev layer
# adding 65 units in first layer and so on . activation is relu
# input shape is 30,1662 for each video i.e 30 frames and 1662 keypoints
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
# return seq as false cuz next is dense layer so not required
model.add(LSTM(64, return_sequences=False, activation='relu'))

# adding 64 units for dense layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# actions is having three values so the actions.shape of [0] is also 3 in shape 
# using softmax so that the values are confined in 0 to 1 the value will sum up and provide 1
model.add(Dense(actions.shape[0], activation='softmax'))


# In[33]:


# eg
eg_res = [.7, 0.2, 0.1]
actions[np.argmax(eg_res)]


# In[34]:


# using the adam optimizer
# categorical_crossentropy for multiclasss classification 
# metrics for evaluation
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[53]:


model.fit(X_train, y_train, epochs=330, callbacks=[tb_callback])
# tensorboard --logdir=.


# In[36]:


model.summary()


# 8. Making the predictions

# In[37]:


res=model.predict(X_test)


# In[38]:


# again the actions with the max value provided by softmax is returned
actions[np.argmax(res[0])]


# In[39]:


actions[np.argmax(y_test[4])]


# Saving weights for future accessability

# In[40]:


model.save('action.h5')


# In[ ]:


del model


# In[41]:


model.load_weights('action.h5')


# Evaluation using Confusion Matrix and Accuracy score

# In[43]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[44]:


yhat = model.predict(X_train)


# In[45]:


# so here we are check the results w.r.t the axis - 1 i.e the row no 1 having the values of actions i.e 3 values
# then converting them in list format and finding the max value
ytrue = np.argmax(y_train, axis=1).tolist()
# one hot encoding
yhat = np.argmax(yhat, axis=1).tolist()


# In[46]:


yhat


# In[47]:


# confution matrix
multilabel_confusion_matrix(ytrue, yhat)


# In[48]:


accuracy_score(ytrue, yhat)


# FINAL Testing in real time

# In[49]:


# for coloring the actions
colors = [(245,117,16), (117,245,16), (16,117,245)]

# results from the model prediction, actions, image from the video, colors from above
def prob_viz(res, actions, input_frame, colors):
    
    output_frame = input_frame.copy()
    # so here prob can be obtained from the softmax from earlier - have 3 values
    for num, prob in enumerate(res):
        # .rectangle for formation of rectangle
        # here the 2nd parameter denotes the position of the color where num can be 0, 1, 2 based on the action and changes the y axis accordingly
        # int prob for x input will help in setting the bar length based on the accuracy of the model prediction and y axis same as above and
        # colors will call the colors function based on the num (of action)
        # -1 for filling up the box
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# In[54]:


# seq will collect the 30 frames for prediction
sequence=[]
# concatinate the seq from history
sentence=[]
predictions=[]
threshold=0.5



cap = cv2.VideoCapture(0)
# Set mediapipe mqodel 
# so we are accesing the mediapipe model using the with mp_holistic.Holistic 
# so how the mediapipe model works is that it actuallly makes an initial detection using the min_detection_confidence 
# then track the key points with min_tracking_confidence=0.5 we can change it as well
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        # for entering the function
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # # Draw landmarks
        # helps in accessing the draw_landmarks func allowing to draw landmarks through mediapipe 
        draw_styled_landmarks(image, results)



        # ==> prediction logic
        # extracting the keypoints for the seq
        keypoints=extract_keypoints(results)
        sequence.append(keypoints)
        # sequence.insert(0,keypoints)

        # here we are collecting the last 30 frames to generate the predictions 
        sequence=sequence[-30:]



        # so if sequence is 30 ie 30th seq then only the prediction value is entered like the if clause below
        if len(sequence)==30:
            # here we are predicting based on the 30 frames and here perticularly np.expand_dims is allowing us to grab the values 
            # which are basically utilized for the perticular action 
            # like for eg we see the shape of the X_train then the original shape is going to be 3 d arr unless we use .tolist and as a result we cannot access 
            # the value present at axis=0 ie for the actions for that perticular seq for this reason np.expand_dims help in accessing the values
            res=model.predict(np.expand_dims(sequence,axis=0))[0]
            # printing using the softmax prediction
            print(actions[np.argmax(res)])
            # appending the black arr
            predictions.append(np.argmax(res))

        # ==> visualization logic
        # so here basically the last 10 values from the prediction arr are being check with the res 
        # providing a more better accuracy for predictions
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        # here in the below line we are checking that the previous word is not the same as of the last word of the sent
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    # so last 5 values will be displayed not resulting in the clutter of words on the screen 
                    sentence = sentence[-5:]



            # ==> Viz probabilities
            # entering the prob func for the better visualization
            image = prob_viz(res, actions, image, colors)



        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[71]:


cap.release()
cv2.destroyAllWindows()

