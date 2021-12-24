# pip3 install numpy seaborn

# example usage : python3 Harry_Plotter.py ./path_to_10_FOLD_X-valid_results/
#                 python3 Harry_Plotter.py ./path_to_10_FOLD_X-valid_results/ cum
#                 python3 Harry_Plotter.py ./path_to_10_FOLD_X-valid_results/ each

import os, sys
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

labels = [
    "dog_bark",
    "children_playing",
    "air_conditioner",
    "street_music",
    "gun_shot",
    "siren",
    "engine_idling",
    "jackhammer",
    "drilling",
    "car_horn",
]

if len(sys.argv) >= 2:
    path = sys.argv[1]
    allConf = sys.argv[2] if (len(sys.argv) >= 3) else "all"
    File = "train_log.txt"
    sumTrix = np.zeros((10, 10), dtype=int)
    xValidResults = []
    for i in range(10):
        xValidResults.append([])
    for root, dirs, files in os.walk(path):
        if File in files:
            paths = os.path.join(root, File)
            with open(paths, "r") as file:
                lines = file.readlines()
                print("\n" + paths)
                strMatrix = ""
                strResult = ""
                for line in lines[38:48]:
                    line = (
                        line.replace("'", "")
                        .replace("\n", "")
                        .replace("]", ";")
                        .replace("[", "")
                        .replace(",", "")
                        .replace(";;", "")
                    )
                    strMatrix += line
                matrix = np.matrix(strMatrix)
                sumTrix = sumTrix + matrix
                
                if allConf == "all" or allConf == "each":
                    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
                    plt.figure(figsize=(10, 7))
                    plt.title(paths)
                    sn.heatmap(df_cm, annot=True)

                cnt = 0
                for line in lines[27:37]:
                    line = line[2:].replace("\n", "").replace(",", "")
                    toPrint = labels[cnt] + " = "
                    print(toPrint + line.rjust(60 - len(toPrint)))
                    xValidResults[cnt].append(float(line))
                    cnt += 1
    print("\n-- Mean Accuracy calculus --")
    cnt = 0
    labelRates = []
    for labelRate in xValidResults:
        val = np.mean(np.asarray(labelRate))
        toPrint = "Mean acc on " + labels[cnt] + " = "
        print(toPrint + str(val).rjust(60 - len(toPrint)))
        labelRates.append(val)
        cnt +=1 
    print("\n\nOverall mean acc " + str(np.sum(np.asarray(labelRates))))

    if allConf == "all" or allConf == "cum":
        print("\n\nSum of every confusion matrix of current folder")
        df_cm = pd.DataFrame(sumTrix, index=labels, columns=labels)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title(path)
    plt.show()
else:
    print(
        "ARGUMENT[1] Missing : Expected Path to 10-FOLD X VALIDATION result dir"
        + " - Example usages underneath : \n"
        + "\t\tpython3 Harry_Plotter.py ./path_to_10_FOLD_X-valid_results/"
        + "\t\tpython3 Harry_Plotter.py ./path_to_10_FOLD_X-valid_results/ cum"
        + "\t\tpython3 Harry_Plotter.py ./path_to_10_FOLD_X-valid_results/ each"
    )