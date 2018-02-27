#!/usr/bin/python3

import csv

with open("../Data/Crimes_-_2015.csv", 'rt') as f:
    reader = csv.reader(f)
    a = []
    i = 0;
    for row in reader:
        a.append (row)
        if (i == 1):
            break
        i += 1

with open ("../Data/fields.txt", "wt") as f:
    for i in range (len (a[0])):
        f.write ("{}:{}\n".format (a[0][i], a[1][i]))

        
