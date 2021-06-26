import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import Decimal
for i in range (1,21):
	j = int(i*5)
	for target in ['dd','lnm']:
		x, y = np.meshgrid(np.linspace(0, 5, 6),np.linspace(0, 10, 11))
		z = pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['AUC'].values
		for item in range (0,len(np.array(z))):
			if np.array(z)[item] == np.max(np.array(z)):
				bestAUC = round(np.array(z)[item],4)
				bestCLASS = pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['Classifier_Method'][item]
				bestFS = pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['Subset'][item]
				bestsp = round(np.array(pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['Specificity'].values)[item],4)
				bestse = round(np.array(pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['Sensitivity'].values)[item],4)
				besttp = pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['TP'][item]
				besttn = pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['TN'][item]
				bestfp = pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['FP'][item]
				bestfn = pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['FN'][item]
				bestacc= round(np.array(pd.read_csv('./metrics/{}_metrics_{}.csv'.format(target,j))['ACC'].values)[item],4)
		z = z.reshape((10,5))
		z_min, z_max = z.max(), z.max()
		fig, ax = plt.subplots()
		c = ax.pcolormesh(x, y, z, cmap='BuPu', vmin=0, vmax=1)
		plt.xticks(np.arange(1,6)-0.5, ('JMI','MRMR','SKB','SP','WLCX'))
		plt.yticks(np.arange(1,11)-0.5, ('ADAC', 'BAGC', 'BNB', 'DTC', 'GNBC', 'KNNC', 'RFC', 'SGDC','SVMC', 'XGBC'))
		plt.xlabel('Feature Selection Method')
		plt.ylabel('Classifier')
		for i in range (0,10):
			for j in range (0,5):
				plt.text(j+0.25,i+0.25,Decimal(z[i][j]).quantize(Decimal("0.00")))
		fig.colorbar(c, ax=ax)
		plt.plot([0.05, 0.95], [1.95, 1.95], color='red', linestyle='-')
		plt.plot([0.05, 0.95], [1, 1], color='red', linestyle='-')
		plt.plot([0.05, 0.05], [1, 1.95], color='red', linestyle='-')
		plt.plot([0.95, 0.95], [1, 1.95], color='red', linestyle='-')
		plt.tight_layout()

		plt.savefig('fig_heatmap_{}_{}.png'.format(target,j))
		plt.savefig('fig_heatmap_{}_{}.pdf'.format(target,j))
		plt.savefig('fig_heatmap_{}_{}.tiff'.format(target,j))
		plt.show()