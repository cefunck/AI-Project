import csv

lista = []
ids = []
ids_por_borrar = []
lista_limpia_oficial = []

with open(r'C:\Users\Cristian\PycharmProjects\AI-Project\input\Input.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        lista.append(row[0].split(';'))

for l in lista:
    if l[1] not in ids:
        ids.append(l[1])

for i in ids:
    borrar = True
    for j in lista:
        if j[1] == i:
            if j[2] != 'ING':
                borrar = False
                break
    if borrar == True:
        ids_por_borrar.append(i)

for l in lista:
    if l[1] in ids_por_borrar:
        lista.remove(l)
lista = lista[1:]

def uniq(lst):
    print(len(lst), "largo de la lista con duplicados")
    return list(set(lst))

for l in lista:
    linea = ";".join(l)+'\r\n'
    lista_limpia_oficial.append(linea)

lista_limpia_oficial = uniq(lista_limpia_oficial)
print(len(lista_limpia_oficial), "largo de la lista sin duplicados")

file = open(r'C:\Users\Cristian\PycharmProjects\AI-Project\input\Output.csv','w')
for i in lista_limpia_oficial:
    file.write(i)