import os
import numpy as np
import pandas as pd
from pathlib import Path 

columns = ['class_id','sample_name']

filepath = Path('data/dataset/BioVid_sequence.csv')  

df = pd.read_csv('data/dataset/BioVid_samples.csv',usecols=columns, sep='\t')

df['num_frames'] = 138
     
df = df[["sample_name","class_id","num_frames"]]
df = df.rename(columns = {'sample_name': 'sequence_name', 'class_id': 'VAS'}, inplace = False)
print(df)
df.to_csv(filepath, index=False)