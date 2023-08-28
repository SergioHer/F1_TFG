import json
import statistics

# Abre el archivo JSON con los datos
with open("global_metrics.json", "r") as json_file:
    data = json.load(json_file)

pilotos = {}
for item in data:
    piloto = item["Piloto"]
    accuracy = item["Accuracy"]
    
    if piloto not in pilotos:
        pilotos[piloto] = []
    
    pilotos[piloto].append(accuracy)

for piloto, accuracies in pilotos.items():
    mean_accuracy = statistics.mean(accuracies)
    std_deviation = statistics.stdev(accuracies)
    
    file_name = f"{piloto}/{piloto}_metrics.txt"
    with open(file_name, "w") as file:
        file.write(f"Piloto: {piloto}\n")
        file.write(f"Media de Precisión: {mean_accuracy:.2f}\n")
        file.write(f"Desviación Estándar: {std_deviation:.2f}\n")