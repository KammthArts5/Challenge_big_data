import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from def_preprocess_data import *
from def_significance import *

#%%import data

data = import_excel_data("factory_process_KC12 KC12_cleaned.xlsx")
Len, dim, Columns = get_metadata(data)
N_data = normalize_data(data)


#%%PCA method

pca = PCA()                                                                 # faire appel à la méthode ACP
Tr_data = pca.fit_transform(N_data)                                         # données transformées (dans la nouvelle base)
val_pr = ((Len-1)/Len)*pca.explained_variance_                              # valeurs propres
vec_pr = pca.components_                                                    # vecteurs propres
Inertie = pca.explained_variance_ratio_                                     # variation liée à chaque axe principal

Inertie_sum = np.copy(Inertie)                                              # variation cumulative
for i in range(1,dim):
    Inertie_sum[i] = Inertie_sum[i] + Inertie_sum[i-1]
    
#%% Synthetize results

pca_results = pd.DataFrame(data=np.vstack((val_pr,Inertie,Inertie_sum)), 
             index=["Valeur propre", "Variation", "Variation cumulative"], columns=["Axe"+str(i+1) for i in range(dim)]).T
Corr_ax_par = corr_axes_param(val_pr, vec_pr, dim, Columns)
fig01 = plot_eigenvalues(val_pr, Inertie_sum, dim)
fig02 = plot_projection(1, 2, 'KC2', data, Tr_data, Corr_ax_par, Inertie, Columns, dim) 
fig03,Corr_par_par = plot_corr_mat(data)

# enregistrer toutes les figures
fig01.savefig('Figures/eigenvalues.jpg',format='jpg', dpi=300)      
fig02.savefig('Figures/projections.jpg',format='jpg', dpi=300)
fig03.savefig('Figures/correlation_matrix.jpg',format='jpg', dpi=300)

#%% Test
