def notaNormalizada(strNota):
    if(strNota == str(0)):
        return "-1"
    print(strNota)
    n = ((float(strNota)-1)/6.0)
    return str(n)

csv = []
csvOficial = []
csvOficial2 = []
file = open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\Output2vSB.csv', 'r')
for i in file.readlines():
    csv.append(i.split(';'))

for i in csv:
    csvOficial.append([])
    for j in i:
        csvOficial[-1].append(notaNormalizada(j))

for l in csvOficial:
    linea = ";".join(l)+'\r\n'
    csvOficial2.append(linea)

file = open(r'C:\Users\MaríaJosé\PycharmProjects\ProyectoIA\AI-Project\input\OFICIAL.csv','w')
for i in csvOficial2:
    file.write(i)

