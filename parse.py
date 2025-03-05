# Nombre de ranks attendu (fourni par la variable 'rank')
rank = 6  # Par exemple, 6 ranks (0 à 5)

# Structures pour stocker les données
value_to_send = [[] for _ in range(rank)]  # Liste de listes pour chaque rank
nb_cells_per_proc = [[] for _ in range(rank)]  # Une seule valeur par rank pour nbCellsPerProc

# Exemple de contenu du fichier (simulé ici, à remplacer par la lecture réelle)
output = """
value_to_send : 686 in rank 0
value_to_send : 1215 in rank 1
value_to_send : 2383 in rank 2
value_to_send : 1728 in rank 3
value_to_send : 1460 in rank 4
value_to_send : 3407 in rank 5
value_to_send : 686 in rank 0
value_to_send : 1215 in rank 1
value_to_send : 2383 in rank 2
value_to_send : 1728 in rank 3
value_to_send : 1460 in rank 4
value_to_send : 3407 in rank 5
iteration 3: t = 0.001953125, dt = 0.00048828125
value_to_send : 686 in rank 0
value_to_send : 1215 in rank 1
value_to_send : 2461 in rank 2
value_to_send : 1738 in rank 3
value_to_send : 1452 in rank 4
value_to_send : 3427 in rank 5
nbCellsPerProc : Proc 0 : 4856 Proc 1 : 4910 Proc 2 : 4859 Proc 3 : 4910 Proc 4 : 4916 Proc 5 : 5025

value_to_send : 686 in rank 0
value_to_send : 1215 in rank 1
value_to_send : 2383 in rank 2
value_to_send : 1728 in rank 3
value_to_send : 1460 in rank 4
value_to_send : 3407 in rank 5
value_to_send : 686 in rank 0
value_to_send : 1215 in rank 1
value_to_send : 2383 in rank 2
value_to_send : 1728 in rank 3
value_to_send : 1460 in rank 4
value_to_send : 3407 in rank 5
iteration 3: t = 0.001953125, dt = 0.00048828125
value_to_send : 686 in rank 0
value_to_send : 1215 in rank 1
value_to_send : 2461 in rank 2
value_to_send : 1738 in rank 3
value_to_send : 1452 in rank 4
value_to_send : 3427 in rank 5
nbCellsPerProc : Proc 0 : 4856 Proc 1 : 4910 Proc 2 : 4859 Proc 3 : 4910 Proc 4 : 4916 Proc 5 : 5025
"""

try:
    with open('output', 'r') as file:
        output = file.read()
except FileNotFoundError:
    print("Erreur : le fichier output.txt n'a pas été trouvé.")
    exit(1)


# Lecture ligne par ligne
lines = output.split('\n')
for line in lines:
    line = line.strip()  # Enlever les espaces au début et à la fin
    if not line:  # Ignorer les lignes vides
        continue
    
    # Parser les lignes avec value_to_send
    if "value_to_send" in line:
        parts = line.split()
        value = int(parts[2])  # La valeur est en position 2
        rank_num = int(parts[-1])  # Le rank est le dernier élément
        value_to_send[rank_num].append(value)  # Ajouter la valeur à la liste du rank
    
    # Parser les lignes avec nbCellsPerProc
    elif "nbCellsPerProc" in line:
        parts = line.split()
        for i in range(rank):
            value = int(parts[5 + 4*i])  # Les valeurs commencent à l'index 3, avec un pas de 2
            nb_cells_per_proc[i].append(value)


print(len(value_to_send[0]))
print(len(nb_cells_per_proc[0]))
# Afficher les résultats
print("Value to send par rank :")
for i in range(rank):
#    value_to_send[i] = value_to_send[i][::3]
    print(f"Rank {i}: {value_to_send[i]}")

print("\nNb Cells Per Proc par rank :")
for i in range(rank):
    print(f"Rank {i}: {nb_cells_per_proc[i]}")


import matplotlib.pyplot as plt
for i in range(rank):
    plt.plot(nb_cells_per_proc[i], label="rank i")
plt.show()

for i in range(rank):
    plt.plot(value_to_send[i], label="rank i")
plt.show()



