import matplotlib.pyplot as plt
import json

with open("results/data.json", 'r') as f:
    data = json.load(f)

unique_nodes = data["unique_nodes"]
total_calls = data["total_calls"]



# Plotting the results
plt.plot(unique_nodes)
plt.xlabel("Iterations (Rounds Searched)", fontsize=12)
plt.ylabel("Unique Information Sets", fontsize=12)

plt.savefig("saves/plot.pdf", format='pdf')
plt.close()