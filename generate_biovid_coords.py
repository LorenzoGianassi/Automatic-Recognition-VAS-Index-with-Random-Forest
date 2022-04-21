from msilib import sequence
import os
import numpy as np
import pandas as pd
from pathlib import Path 

path = 'BioVid_dataset/' # Path the landmark folder containig the dataset
subjects = [name for name in os.listdir(path)] # Retrieve all the subjects in the dataset

filepath = Path('data/dataset/BioVid_coords.csv')  
result = pd.DataFrame([])

sequences_indx = 0.0
count = 0

for i, sub in enumerate(subjects):
    
    if sub == '102309_m_61': # Subject to ignore
        continue
    
    path_sequences = path + sub + '/'
    sequences = [name for name in os.listdir(path_sequences)] # Retrieve all the sequences of the current subject

    for seq in sequences:
        path_frame = path_sequences + seq + '/'
        frames = [name for name in os.listdir(path_frame)] # Retrieve all the frames of the current sequence
        for k, frame in enumerate(frames):
            data = np.load(path_frame + frame) # Read the data from the *.npy files (size is (68,2))
            df = pd.DataFrame(data)
            df = pd.DataFrame(np.expand_dims(df.values.reshape(-1),axis=1))
            df = df.transpose()
            df = df.shift(periods=1, fill_value=sequences_indx, axis="columns")
            
            result = result.append(df)         
            
            count += 1
            #print(count)
        sequences_indx += 1
    print(sub)
result.to_csv(filepath, index=False)        


            
            
  