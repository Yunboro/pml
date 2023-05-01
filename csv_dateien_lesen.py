import csv
import cv2

from torch.utils.data import random_split

#STOP 14 -> 0
#Rechts 33 -> 1
#Links 34 -> 2
#Kreis 40 -> 3


def load():
    
    temp = []                 # # Example: [(152,2), (152,2) ...] [(Bild_Zahl, Klasse), ..., (Bild_Zahl_n, Klasse_n)]  n = [0,1,2,3,4]

    for zahl in range(5): # 0,1,2,3,4
    
        filename = "C:/gitdir/machine-learning-project/" + str(zahl) + "/" + str(zahl) + ".csv"
        
        with open(filename, "r", newline='') as csv_bild:
            
            reader = csv.reader(csv_bild, delimiter=';') # Each row from the csv_bild is returned as a list of strings
            
            for zeile in reader:
                
                # Form tuple(String Dateiname, Klasse als int)
                a = (cv2.imread(zeile[1]+"/"+zeile[0]), int(zeile[1]))
                temp.append(a) 
                
    
    return temp

if __name__ == '__main__':
    
    print(load())





    
    
    # Assuming your training dataset is stored in the variable `dataset` with 100% of the data
    
    
    
    #train_size = int(0.8 * len(dataset))  # 80% of the data will be used for training
    #val_size = len(dataset) - train_size  # Remaining 20% of the data will be used for validation
    
    # Use random_split to split the dataset into training and validation subsets
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
