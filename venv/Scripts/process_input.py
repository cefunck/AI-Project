import random as rand
import csv

lista = []
listaInfo = []
ids = []
ids_por_borrar = []
lista_limpia_oficial = []
id_oficiales = []
labels_oficiales = []
semestres = []
ramos = []
matriz = [[]]
id_icc = []
id_ici = []
id_ico = []
id_ice = []

def uniq(lst):
    bla = []
    for l in lst:
        bla.append(";".join(l))
    bla = list(set(bla))
    lst = []
    for l in bla:
        lst.append(l.split(";"))
    return lst

file = open(r'C:\Users\Cristian\PycharmProjects\AI-Project\input\base_dato_muestra.csv', 'r')
for i in file.readlines():
    lista.append(i.split(';'))

file = open(r'C:\Users\Cristian\PycharmProjects\AI-Project\input\Input.csv', 'r')
for i in file.readlines():
    listaInfo.append(i.split(';'))
listaInfo = listaInfo[1:]

for l in lista:
    if l[1] not in ids:
        ids.append(l[1])

for i in ids:
    borrar = True
    for j in lista:
        if j[1] == i and j[2] != 'ING':
            borrar = False
            break
    if borrar == True:
        ids_por_borrar.append(i)
for l in lista:
    if l[1] in ids_por_borrar:
        lista.remove(l)
lista = lista[1:]

for i in ids:
    borrar = True
    for j in listaInfo:
        if j[1] == i and j[2] != 'ING':
            borrar = False
            break
    if borrar == True:
        ids_por_borrar.append(i)
for l in listaInfo:
    if l[1] in ids_por_borrar:
        listaInfo.remove(l)




for i in lista:
    id_oficiales.append(i[1])
    semestres.append(int(i[0]))
    ramos.append(i[4])
id_oficiales = list(set(id_oficiales))
#semestres = sorted(list(set(semestres)))
#ramos = sorted(list(set(ramos)))

for i in listaInfo:
    semestres.append(int(i[0]))
    ramos.append(i[4])
id_oficiales = list(set(id_oficiales))
semestres = sorted(list(set(semestres)))
ramos = sorted(list(set(ramos)))
#print(semestres)


for id in id_oficiales:
    info = []
    for l in lista:
        if l[1] == id:
            info.append(l)
    definitiva = []
    for inf in info:
        if definitiva == []:
            definitiva = inf
        elif (int(definitiva[0]) < int(inf[0])):
            definitiva = inf
    if definitiva[2] == "INGC":
        id_icc.append(definitiva[1])
    elif definitiva[2] == "INGE":
        id_ice.append(definitiva[1])
    elif definitiva[2] == "INGO":
        id_ico.append(definitiva[1])
    elif definitiva[2] == "INGI":
        id_ici.append(definitiva[1])

id_oficiales = []

for i in id_icc:
    id_oficiales.append(i)
    labels_oficiales.append('[1 0 0 0]')

for i in id_ice:
    id_oficiales.append(i)
    labels_oficiales.append('[0 1 0 0]')

for i in id_ico:
    id_oficiales.append(i)
    labels_oficiales.append('[0 0 1 0]')

for i in id_ici:
    id_oficiales.append(i)
    labels_oficiales.append('[0 0 0 1]')

#print("icc: ",len(id_icc)," ice: ",len(id_ice)," ico: ",len(id_ico)," ici: ",len(id_ici))

for i in semestres:
    #print(ramos)
    for j in ramos:
        matriz[-1].append(j)

id_icc_oficiales = []
id_ice_oficiales = []
id_ico_oficiales = []
id_ici_oficiales = []

for i in range((min(len(id_icc),len(id_ice),len(id_ici),len(id_ico)))):
    idx_icc = rand.randint(0, len(id_icc)-1)
    idx_ice = rand.randint(0, len(id_ice)-1)
    idx_ico = rand.randint(0, len(id_ico)-1)
    idx_ici = rand.randint(0, len(id_ici)-1)
    id_icc_oficiales.append(id_icc.pop(idx_icc))
    id_ice_oficiales.append(id_ice.pop(idx_ice))
    id_ico_oficiales.append(id_ico.pop(idx_ico))
    id_ici_oficiales.append(id_ici.pop(idx_ici))

#print("icc oficial: ",len(id_icc_oficiales)," ice oficial: ",len(id_ice_oficiales)," ico oficial: ",len(id_ico_oficiales)," ici oficial: ",len(id_ici_oficiales))

lista = uniq(lista)

for id in id_oficiales: #aqui poner la lista que correremos
    matriz.append([])
    for s in semestres:
        for r in ramos:
            agregarCero = True;
            for line in lista:
                if line[0] == str(s):
                    if line[1] == id:
                        if line[4] == r:
                            matriz[-1].append(str(line[5][:-1]))
                            agregarCero = False
            if agregarCero:
                matriz[-1].append(str(0))

for l in matriz[1:]:

    linea = ";".join(l)+'\r\n'
    lista_limpia_oficial.append(linea)

file = open(r'C:\Users\Cristian\Desktop\EjemploProyectoFinal.csv','w')
for i in lista_limpia_oficial:
    file.write(i)


lista_limpia_oficial_2 = []
for l in labels_oficiales:
    linea = " ".join(l)+'\r\n'

    lista_limpia_oficial_2.append(linea)

file = open(r'C:\Users\Cristian\PycharmProjects\AI-Project\input\Labels_oficiales_cote.csv','w')
for i in lista_limpia_oficial_2:
    file.write(i)

