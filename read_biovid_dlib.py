import os
import json
import pandas as pd
from pathlib import Path 
from json import JSONEncoder
import numpy as np

path_landmarks = 'C:/Users/gigli/Downloads/Landmarks-Biovid-Dlib/landmarkall/' # Path to Biovid-landmarks-labels.zip extracted
subjects = [name for name in sorted(os.listdir(path_landmarks))]  # List of subjects



# FIRST correspond to first element of dataset used to costruct training or test file 
# LAST correspond to last element of dataset 

FIRST = 0
LAST = len(subjects)
print(subjects)
print(LAST)
file_name = 'dataset_final_' + str(LAST-FIRST) + '.csv'

filepath = Path('BioVid_Dataset/result.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  

path_save = '/path_save/' + file_name # Path where save csv file


lst = []
element = []

for i in range(FIRST,LAST):

    subject = subjects[i]
    sequences_path = path_landmarks + subject + '/'
    sequences = [name for name in sorted(os.listdir(sequences_path))]  # List of sequences


    for num_seq in range(0,len(sequences)):

        element.append(i)
        element.append(num_seq)

        print(element)

        seq = sequences[num_seq]
        frame = sequences_path + seq
        # Open JSON file and read coordinates
        with open(frame, 'r') as f:
    
            json_dict = np.load(f)
            if len(json_dict['people']) != 0:
                landmarks = json_dict['people'][0]['face_keypoints_2d']
            else:
                print(subject , " - " , num_seq)

            # Get x and y coordinates (third component is a confidence score)
            ll_x = [landmarks[x] for x in range(0, len(landmarks), 3)]
            ll_y = [landmarks[x] for x in range(1, len(landmarks), 3)]
            confidence = [landmarks[x] for x in range(2, len(landmarks), 3)]

            for k in ll_x:
                element.append(k)
            for k in ll_y:
                element.append(k)

        lst.append(element)
        element = []


col = range(0,142)
df = pd.DataFrame(lst,columns=col)
print(df)
#df.to_csv(index=False)   
df.to_csv(filepath, index=False)

