import os
d = 'C:/Users/gigli/Downloads/Landmarks-Biovid-Dlib/landmarks'
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
print(subdirs)