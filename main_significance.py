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
fig022 = plot_projection(1, 2, 'KC12', data, Tr_data, Corr_ax_par, Inertie, Columns, dim) 
fig03,Corr_par_par = plot_corr_mat(data)

# enregistrer toutes les figures
fig01.savefig('Figures/eigenvalues.jpg',format='jpg', dpi=300)      
fig02.savefig('Figures/projectionsKC2.jpg',format='jpg', dpi=300)
fig022.savefig('Figures/projectionsKC12.jpg',format='jpg', dpi=300)
fig03.savefig('Figures/correlation_matrix.jpg',format='jpg', dpi=300)

#%% Export results

writer = pd.ExcelWriter('Résultats_KC2.xlsx', engine='xlsxwriter')

data.to_excel(writer, sheet_name='données initiales')

N_data = pd.DataFrame(data=N_data, columns=Columns)
N_data.to_excel(writer, sheet_name='données normalisées')

Tr_data = pd.DataFrame(data=Tr_data, columns=["Axe"+str(i+1) for i in range(dim)])
Tr_data.to_excel(writer, sheet_name='données transformées')

pca_results.to_excel(writer, sheet_name='Valeurs propres et Inertie')
worksheet = writer.sheets['Valeurs propres et Inertie']
worksheet.insert_image('G1', 'Figures/eigenvalues.jpg')

Corr_ax_par.to_excel(writer, sheet_name='cercle de corrélations', startcol=26)
worksheet = writer.sheets['cercle de corrélations']
worksheet.insert_image('A1', 'Figures/projectionsKC2.jpg')
worksheet.insert_image('A1', 'Figures/projectionsKC12.jpg')

Corr_par_par.to_excel(writer, sheet_name='corrélations entre paramètres', startcol=19)
worksheet = writer.sheets['corrélations entre paramètres']
worksheet.insert_image('A1', 'Figures/correlation_matrix.jpg')

writer.save()
