import csv


lista = []
ids = []
ids_por_borrar = []

with open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\Input.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        lista.append(row[0].split(';'))



for l in lista:

    if l[1] not in ids:
        ids.append(l[1])


for i in ids:
    borrar = True
    especialidades = []
    for j in lista:

        if j[1] == i:
            especialidades.append(j[2])
            if j[2] != 'ING':
                borrar = False
                break
    if borrar == True:
        ids_por_borrar.append(i)


for l in lista:
    if l[1] in ids_por_borrar:
        lista.remove(l)

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

lista_limpia = sort_and_deduplicate(lista)

for l in lista_limpia:
    print(l)

#with open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\Output.csv', 'w') as myfile:
 #   wr = csv.writer(myfile, lineterminator= '\n')
  #  wr.writerow(lista_limpia)