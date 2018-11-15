import csv


lista = []
ids = []
ids_por_borrar = []

with open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\Input.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        lista.append(row[0].split(';'))


print(lista)
for l in lista:

    if l[1] not in ids:
        ids.append(l[1])


for i in ids:
    borrar = True
    especialidades = []
    for j in lista:

        if j[1] == i:


            especialidades.append(j[2])


    if borrar == True:
        ids_por_borrar.append(i)





