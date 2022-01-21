import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('assets/parameter_size500.csv')[15::]
size = 500

cprrcoef = np.corrcoef(df['talerate_preds_base'],df['foolingrate_preds_rate'])[0,1]
plt.show()
for index, d in df.iterrows():
        plt.plot(d['talerate_preds_base'],d['foolingrate_preds_rate'],'o')
        plt.annotate(d['model_name'], xy=(d['talerate_preds_base'],d['foolingrate_preds_rate']))

plt.xlabel('Tale rate')
plt.ylabel('Foolingrate')
plt.title(f'cprrcoef: {cprrcoef}')
plt.subplots_adjust(left=0.1, right=0.85)
plt.savefig(f'assets/talerate_foolingrate_corrcoef_exef_size{size}.png')
plt.close()

cprrcoef = np.corrcoef(df['talerate_preds_base'],df['clean_acc'])[0,1]
plt.show()
for index, d in df.iterrows():
        plt.plot(d['talerate_preds_base'],d['clean_acc'],'o')
        plt.annotate(d['model_name'], xy=(d['talerate_preds_base'],d['clean_acc']))

plt.xlabel('Tale rate')
plt.ylabel('Accuracy')
plt.title(f'cprrcoef: {cprrcoef}')
plt.subplots_adjust(left=0.15, right=0.85)
plt.savefig(f'assets/talerate_acc_corrcoef_exef_size{size}.png')
plt.close()

cprrcoef = np.corrcoef(df['clean_acc'],df['foolingrate_preds_rate'])[0,1]
plt.show()
for index, d in df.iterrows():
        plt.plot(d['clean_acc'],d['foolingrate_preds_rate'],'o')
        plt.annotate(d['model_name'], xy=(d['clean_acc'],d['foolingrate_preds_rate']))

plt.xlabel('Accuracy')
plt.ylabel('Foolingrate')
plt.title(f'cprrcoef: {cprrcoef}')
plt.subplots_adjust(left=0.1, right=0.85)
plt.savefig(f'assets/acc_foolingrate_corrcoef_exef_size{size}.png')
plt.close()