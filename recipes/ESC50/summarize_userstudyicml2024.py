import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

df = pd.read_csv('mushra_dd.csv')

users = df.email.unique().tolist()

print('We have ', len(users), 'users')

users_filtered = df[df.email != users[0]]
users_filtered = users_filtered[users_filtered.email != 'teo']
users_filtered = users_filtered[users_filtered.email != 'LC']
users_filtered = users_filtered[users_filtered.email != 'dtgfghbjo']

users = users_filtered.email.unique().tolist()
print('We have ', len(users), 'users')


# users_filtered = users_filtered[df.email == 'Mirco']

average_scores = users_filtered.groupby(['age', 'gender']).trial_id.apply(lambda x:x.astype('float').mean())
all_scores = users_filtered.groupby(['age', 'gender']).trial_id.apply(lambda x:np.concatenate([x.astype('float')]))

companion_average = average_scores.iloc[:24]
news_average = average_scores.iloc[24:]

companion_all_average = companion_average.groupby('gender').mean()
news_all_average = news_average.groupby('gender').mean()

print(companion_all_average)
print(news_all_average)

#import pdb; pdb.set_trace()

w = 4
color = ['red', 'blue', 'yellow', 'cyan', 'magenta', 'green']
labels = ['L-MAC', 'L-MAC-FT1', 'L-MAC-FT2', 'L2I-R1', 'L2I', 'Ref']

fs = 14
sh = 0.4
plt.figure(figsize=[12, 4], dpi=100)
plt.subplot(121)
for j in range(4):
    for n, nl in enumerate([0, 1, 2, 4]):
        if j == 0:
            plt.bar(j*w + 0.4*n, companion_average.values[j*6 + nl], width=0.4, color=color[n], label=labels[nl]) #color=color[n])
        else:
            plt.bar(j*w + 0.4*n, companion_average.values[j*6 + nl], width=0.4, color=color[n],) #color=color[n])

plt.xticks([0*w +sh, 1*w +sh, 2*w+sh, 3*w+sh], ['Recording 1', 'Recording 2', 'Recording 3', 'Recording 4'] ) 
plt.ylabel('MOS', fontsize=fs)
            #plt.bar(j*w + 0.4*n, companion_average.values[j*6 + n], width=0.4, color=color[n]) #color=color[n])

plt.legend(fontsize=fs, loc='lower right')

plt.title('L2I Recordings', fontsize=fs)
plt.tight_layout()
#plt.savefig('companion_ustudy.pdf')


##########
#plt.figure(figsize=[12, 5], dpi=100)

color = ['red', 'blue', 'yellow', 'cyan', 'magenta', 'green']
labels = ['L-MAC', 'L2I', 'L2I-R2', 'L-MAC-FT1', 'L-MAC-FT2', 'Ref']


sh = 0.4
#plt.figure()
plt.subplot(122)
for j in range(5):
    for n, nl in enumerate([0, 3, 4, 1]):
        if j == 0:
            plt.bar(j*w + 0.4*n, news_average.values[j*6 + nl], width=0.4, color=color[n], label=labels[nl]) #color=color[n])
        else:
            plt.bar(j*w + 0.4*n, news_average.values[j*6 + nl], width=0.4, color=color[n]) #color=color[n])

plt.xticks([0*w +sh, 1*w +sh, 2*w+sh, 3*w+sh, 4*w + sh], ['Recording 1', 'Recording 2', 'Recording 3', 'Recording 4', 'Recording 5'] ) 
plt.ylabel('MOS', fontsize=fs)

plt.legend(fontsize=fs, loc='lower right')
plt.title('New Random Recordings', fontsize=fs)
plt.tight_layout()

# plt.savefig('ustudy.pdf')
