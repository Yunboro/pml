import csv

#STOP 14 -> 0
#Rechts 33 -> 1
#Links 34 -> 2
#Kreis 40 -> 3


def load():
    
    temp = []                 # # Example: [(152,2), (152,2) ...] [(Bild_Zahl, Klasse), ..., (Bild_Zahl_n, Klasse_n)]  n = [0,1,2,3,4]

    for zahl in range(5): # 0,1,2,3,4
    
        filename = "C:/gitdir/machine-learning-project/" + str(zahl) + "/" + str(zahl) + ".csv"
        
        with open(filename, "r", newline='') as csv_bild:
            
            reader = csv.reader(csv_bild, delimiter=';') # Each row rad from the csv_bild is returned as a list of strings
            
            for zeile in reader:
                
                # Form tuple(String Dateiname, Klasse als int)
                a = (zeile[0], int(zeile[1]))
                temp.append(a) 
                
    
    return temp

if __name__ == '__main__':
    
    print(load())


