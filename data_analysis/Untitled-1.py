import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import safe_mpc.plut as plut
#folder = '/home/utente/Documents/Optim/mpc-dock-default2/safe-mpc/datauniform/2024-07-27_15-43-42PARTIAL_CONDENSING_HPIPM100800parallelnewx0'
#folder = '/home/utente/Documents/Optim/mpc-dock-default2/safe-mpc/data/2024-07-27_15-27-16PARTIAL_CONDENSING_HPIPM100800regedingnewx0'
#folder = '/home/utente/Documents/Optim/mpc-dock-default2/safe-mpc/data/2024-07-29_10-21-42PARTIAL_CONDENSING_HPIPM100800'
folder= '/home/utente/Documents/Optim/dat_opt/nocorr'
#folder= '/home/utente/Documents/Optim/dat_opt/2024-07-29_10-21-42PARTIAL_CONDENSING_HPIPM100800'

#folder= '/home/utente/Documents/Optim/DATI/datix0old/old/parallelCheckSafeIntGuessCorrection5e-4'
data = {}
for i in os.listdir(folder):
    if i[-4:]=='.pkl':
        with open(folder+'/'+i,'rb') as f:
            data[i[:-4]] = pickle.load(f)
            
err = [item for sublist in data['errors'] for item in sublist]
err=np.array(err).flatten()
err=err[~np.isnan(err)]
print(err.shape)
bound_l = 1e-3
bound_h = 10
err = err[err >= bound_l]
err = err[err<=bound_h]


#bin_edges = np.linspace(bound_l,bound_h, num=10)
plt.hist(err,bins=10, color='blue', alpha=0.5, edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Error')

plt.savefig('errors')
# Display the plot
plt.show()
