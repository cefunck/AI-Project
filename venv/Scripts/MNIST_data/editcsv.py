import csv
import random as rand

with open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\OFICIAL.csv') as input, open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\Output2vSB.csv', 'w', newline='') as output:
     writer = csv.writer(output)
     for row in csv.reader(input):
         if any(field.strip() for field in row):
             writer.writerow(row)

with open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\Labels_oficiales_cote.csv') as input, open(
        r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\Labels_oficiales_cote_definitivo.csv', 'w', newline='') as output:
    writer = csv.writer(output)
    for row in csv.reader(input):
        if any(field.strip() for field in row):
            writer.writerow(row)
