#!/usr/bin/python3

import csv

with open("../Data/IUCR.csv", 'rt') as f:
    reader = csv.reader(f)

    a = {}
    i = 0
    j = 11
    for row in reader:
        if (i == 0):
            i += 1
            continue
        a[row[0]] = j
        j += 1

    print (a)
