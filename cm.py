import numpy as np
from sklearn.metrics import auc,roc_curve,confusion_matrix,roc_auc_score
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

label = 	
result = 
mat = confusion_matrix(label, result,labels=[1,0])
plt.figure(figsize=(5,5))
p1 = sn.heatmap(mat, annot=True, annot_kws={'color':'k','size':15},
				cmap = 'Oranges',vmin = 0, vmax=len(label), fmt='g',cbar=False) 
cb = p1.figure.colorbar(p1.collections[0]) 
cb.ax.tick_params(labelsize=10) 
p1.set_xticklabels([1,0],fontsize=15)
p1.set_yticklabels([1,0],fontsize=15)
p1.set_xlabel('Predicted',fontsize=15)
p1.set_ylabel('Actual',fontsize=15)
p1.xaxis.set_ticks_position('top')
p1.xaxis.set_label_position('top')
plt.tight_layout()
# plt.show()
plt.savefig('fig_cm.png')
plt.savefig('fig_cm.pdf')
plt.savefig('fig_cm.tiff')
plt.close()