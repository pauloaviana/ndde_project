import numpy as np


# Set code to run here

def csv_reader(path,filename):
    file_ID = path+"/"+ filename
    matrix = np.loadtxt(open(file_ID, "rb"), delimiter=",",dtype = int)
    return matrix
