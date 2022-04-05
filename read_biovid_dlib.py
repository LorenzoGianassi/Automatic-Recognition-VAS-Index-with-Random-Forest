import os
import numpy as np
import pandas as pd
from pathlib import Path 

path = 'data/dataset/prova/' # Path the landmark folder containig the dataset
subjects = [name for name in os.listdir(path)] # Retrieve all the subjects in the dataset

filepath = Path('BioVid_Dataset/result.csv')  

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
            print("Data: ",df)
            df.to_csv(filepath, index=False)
  