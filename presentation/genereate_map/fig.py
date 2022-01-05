from os import sep
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 30})
plt.rcParams["figure.figsize"] = (15,9)

data = pd.read_csv("run.csv", sep=",")
print(data)
data["Value"].plot()
plt.xlabel('Epochs')
plt.ylabel('mAP 0.5')
plt.tight_layout()
plt.savefig("../fig/map.pdf")
# plt.show()