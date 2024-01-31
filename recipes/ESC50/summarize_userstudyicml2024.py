import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

df = pd.read_csv('mushra.csv')

users = df.email.unique().tolist()

users_filtered = df[df.email != users[0]]
# users_filtered = users_filtered[df.email == 'Mirco']

average_scores = users_filtered.groupby(['age', 'gender']).trial_id.apply(lambda x:x.astype('float').mean())
all_scores = users_filtered.groupby(['age', 'gender']).trial_id.apply(lambda x:np.concatenate([x.astype('float')]))

companion_average = average_scores.iloc[:24]
news_average = average_scores.iloc[24:]

companion_all_average = companion_average.groupby('gender').mean()
news_all_average = news_average.groupby('gender').mean()

w = 4
color = ['red', 'blue', 'yellow', 'cyan', 'magenta', 'green']
labels = ['L-MAC', 'L-MAC-FT1', 'L-MAC-FT2', 'L2I-R1', 'L2I-R2', 'Ref']

plt.figure()
for j in range(4):
    for n in range(6):
        if j == 0:
            plt.bar(j*w + 0.4*n, companion_average.values[j*6 + n], width=0.4, color=color[n], label=labels[n]) #color=color[n])
        else:
            plt.bar(j*w + 0.4*n, companion_average.values[j*6 + n], width=0.4, color=color[n]) #color=color[n])


plt.legend()
plt.savefig('companion_ustudy.png')

plt.figure()

color = ['red', 'blue', 'yellow', 'cyan', 'magenta', 'green']
labels = ['L-MAC', 'L2I-R1', 'L2I-R2', 'L-MAC-FT1', 'L-MAC-FT2', 'Ref']

plt.figure()
for j in range(5):
    for n in range(6):
        if j == 0:
            plt.bar(j*w + 0.4*n, news_average.values[j*6 + n], width=0.4, color=color[n], label=labels[n]) #color=color[n])
        else:
            plt.bar(j*w + 0.4*n, news_average.values[j*6 + n], width=0.4, color=color[n]) #color=color[n])

plt.legend()
plt.savefig('news_ustudy.png')
