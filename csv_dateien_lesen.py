import csv



def make_a_list():
    
    temp = []                 # # Example: [(152,2), (152,2) ...] [(Bild_Zahl, Klasse), ..., (Bild_Zahl_n, Klasse_n)]  n = [0,1,2,3,4]

    for zahl in range(5): # 0,1,2,3,4
    
        filename = "C:/Users/username/dataset/" + str(zahl) + "/" + str(zahl) + ".csv"
        
        with open(filename, "r", newline='') as csv_bild:
            
            reader = csv.reader(csv_bild, delimiter=';') # Each row rad from the csv_bild is returned as a list of strings
            
            for zeile in reader:
                
                # Muss der Inhalt Integer sein??
                
                zeile_integer = tuple([int(e) for e in zeile]) # Example: (152,2) (Bild_Zahl, Klasse)
                
                temp.append(tuple(zeile_integer)) 
            
    
    return temp




