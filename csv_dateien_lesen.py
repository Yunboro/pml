import csv
import cv2

import random

from torch.utils.data import random_split

#STOP 14 -> 0
#Rechts 33 -> 1
#Links 34 -> 2
#Kreis 40 -> 3




def load():
    
    dataset = []                 # [(file_name_1,label), (file_name_2,label), (file_name_3,label)  ...]  label = [0,1,2,3,4]

    for zahl in range(5): # 0,1,2,3,4
    
        filename = "C:/Users/jorge/Desktop/Uni_nichtloeschen/hm/SS2023/Projekt_ML/machine-learning-project/" + str(zahl) + "/" + str(zahl) + ".csv"
        
        with open(filename, "r", newline='') as csv_bild:
            
            reader = csv.reader(csv_bild, delimiter=';') # Each row from the csv_bild is returned as a list of strings
            
            for zeile in reader:
                
                # Form tuple(String Dateiname, Klasse als int)
                a = (zeile[1]+"/"+zeile[0], int(zeile[1]))
                dataset.append(a) 
        
            
        
    # Shuffle the dataset randomly
    random.shuffle(dataset)     
        
        
    # Compute the sizes of the training and validation subsets
        
    train_size = int(0.8 * len(dataset))  # 80% of the data will be used for training
    val_size = len(dataset) - train_size  # Remaining 20% of the data will be used for validation
        
    # Use random_split to split the dataset into training and validation subsets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    
    
    # Form list of tuples [([[R1_image_1,G1_image_1,B1_image_1],[R2_image_1,G2_image_1,B2_image_1],...], label), ([[R1_image_2,G1_image_2,B1_image_2],[R2_image_2,G2_image_2,B2_image_2],...], label) ...]
    
    
    # cv2.imread(element[0], cv2.IMREAD_GRAYSCALE)   Is the color important?? If not, we can process the image with one channel
    train_dataset_final = [(cv2.imread(element[0]), element[1]) for element in train_dataset] 
    
    
    
    # cv2.imread(element[0], cv2.IMREAD_GRAYSCALE)   Is the color important?? If not, we can process the image with one channel
    val_dataset_final = [(cv2.imread(element[0]), element[1]) for element in train_dataset]
    

        
    return train_dataset_final, val_dataset_final



"""
def load():
    
    temp = []                 # # Example: [(152,2), (152,2) ...] [(Bild_Zahl, Klasse), ..., (Bild_Zahl_n, Klasse_n)]  n = [0,1,2,3,4]

    for zahl in range(5): # 0,1,2,3,4
        
    
        filename = "C:/Users/jorge/Desktop/Uni_nichtloeschen/hm/SS2023/gitdir/machine-learning-project/" + str(zahl) + "/" + str(zahl) + ".csv"
        
        with open(filename, "r", newline='') as csv_bild:
            
            
            reader = csv.reader(csv_bild, delimiter=';') # Each row from the csv_bild is returned as a list of strings
            
            for zeile in reader:
                
                
                # Form tuple(String Dateiname, Klasse als int)
                a = (cv2.imread(zeile[1]+"/"+zeile[0]), int(zeile[1]))
                temp.append(a)        
   
    return temp
"""




if __name__ == '__main__':
    
    train_dataset_final = load()[0]
    
    #print(train_dataset_final)
    print(train_dataset_final[0]) # Ein Tuple (Bild_pixel, label)
    print(train_dataset_final[100][0].shape)
    
    

