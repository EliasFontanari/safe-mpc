# %%

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# %%
#folder = '/home/utente/Documents/Optim/mpc-dock-default2/safe-mpc/datauniform/2024-07-27_15-43-42PARTIAL_CONDENSING_HPIPM100800parallelnewx0'
#folder = '/home/utente/Documents/Optim/mpc-dock-default2/safe-mpc/data/2024-07-27_15-27-16PARTIAL_CONDENSING_HPIPM100800regedingnewx0'
#folder = '/home/utente/Documents/Optim/mpc-dock-default2/safe-mpc/data/2024-07-29_10-21-42PARTIAL_CONDENSING_HPIPM100800'
folder= '/home/utente/Documents/Optim/DATI/Datix0new/parallelcheckintegrate'
#folder= '/home/utente/Documents/Optim/DATI/Datix0new/ParallelCheckIntNegativeJmp-5'
data = {}
for i in os.listdir(folder):
    if i[-4:]=='.pkl':
        with open(folder+'/'+i,'rb') as f:
            data[i[:-4]] = pickle.load(f)

# %%
folder2= '/home/utente/Documents/Optim/DATI/Datix0new/recedingNegativeJmp-21'
data2 = {}
for i in os.listdir(folder2):
    if i[-4:]=='.pkl':
        with open(folder2+'/'+i,'rb') as f:
            data2[i[:-4]] = pickle.load(f)

# %%
os.listdir(folder)
with open(folder+'/coreused.pkl','rb') as f:
    ata = pickle.load(f)


# %%
data.keys()

# %%
err = [item for sublist in data['errors'] for item in sublist]
err=np.array(err).flatten()
err=err[~np.isnan(err)]
err = err[err > 1e-4]

plt.hist(err,density=True, bins=100, color='blue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('ERRORS')
 
# Display the plot
plt.show()

np.max(err)

# %%
horizon_safe = []
for i in range(len(data['safehor_hist'])):
    for j in range(len(data['safehor_hist'][i])):
        horizon_safe.append(data['safehor_hist'][i][j])
horizon_safe=np.array(horizon_safe)

horizon_safe_rec =[]
for i in range(len(data2['safehor_hist'])):
    for j in range(len(data2['safehor_hist'][i])):
        horizon_safe_rec.append(data2['safehor_hist'][i][j])
horizon_safe_rec=np.array(horizon_safe_rec)


plt.hist(horizon_safe[horizon_safe<32],density=False, bins=30,color='blue', alpha=.5, edgecolor='black')
plt.hist(horizon_safe_rec[horizon_safe_rec<32],density=False, bins=30,color='yellow' ,alpha=.5,edgecolor='black')

 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('safe horizon [steps]')
 
# Display the plot
plt.show()


# %%
print(np.mean(horizon_safe_rec))
print(np.mean(horizon_safe))

# %%
data['x_u']['x_sim'][0][-1]
def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.array([1,0,0,1,0,0])
    return np.linalg.norm(np.multiply(mask, x - np.array([3.876991,0,0,0,0,0]))) < 1e-3 

# %%
succes_par, succes_rec = [],[]
for i in range(len(data['x_u']['x_sim'])):
    succes_par.append(convergenceCriteria(data['x_u']['x_sim'][i][-1]))
    succes_rec.append(convergenceCriteria(data2['x_u']['x_sim'][i][-1]))

plt.figure()
plt.title('Problems solved parallel')
plt.grid(True)
plt.plot(succes_par,label='parallel')
plt.plot(succes_rec,label='receding')
plt.legend()
plt.show()
#plt.close()

# plt.figure()
# plt.title('Problems solved receding')
# plt.grid(True)
# plt.plot(succes_rec)
# plt.show()
# plt.close()



# %%
n=0
plotted=data['safehor_hist'][n]

plt.figure()
plt.title(f'Safe horizon history problem: {n}')
plt.grid(True)
plt.plot(np.arange(0,len(plotted))*5e-3,plotted)
plt.show()
plt.close()

plotted=data2['safehor_hist'][n]

plt.figure()
plt.title(f'Safe horizon history problem: {n}')
plt.grid(True)
plt.plot(np.arange(0,len(plotted))*5e-3,plotted)
plt.show()
plt.close()

# %%
for i in range(len(data['safehor_hist'])):
    minimumhor = min(data['safehor_hist'][i])
    if minimumhor<10:
        print(f'problem {i} minimum {minimumhor}')

# %%
mean_error_jump  = [sublist if sublist else [0] for sublist in data['error_jump']]
for i in range(len(mean_error_jump)):
   a = np.array(mean_error_jump[i])
   a = a[~np.isnan(a)]
   mean_error_jump[i] = np.mean(a)


plt.figure()
plt.title('Mean error vs jump')
plt.grid(True)
plt.plot(mean_error_jump)
plt.show()
plt.close()

# %%
len(mean_error_jump)

# %%
n_jump=2
err_jump=np.array(data['error_jump'][n_jump]).flatten()
err_jump=err_jump[~np.isnan(err_jump)]
err_jump = err_jump[err_jump > 1e-5]

plt.hist(err_jump, bins=100, color='blue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('ERRORS')
 
# Display the plot
plt.show()


# più errori perchè più step affrontati

# %%
jumps = np.array(data['jumps']) 
#jumps=jumps[jumps>0]
plt.hist(jumps,density=True, bins=30,range=[-2, 30], color='blue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Jumps frequency')
plt.grid(True)
 
# Display the plot
plt.show()

max(jumps)
np.mean(jumps)

# %%
core = np.array(data['coreused'])
plt.hist(core[core<=36], bins=35, color='blue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Core used frequency')
 
# Display the plot
plt.show()

min(core)

# %%
data['coreused']

# %%
xu = data['x_u']
x0=[]
x=xu['x_sim']
tot=0
for i in range(len(x)):
    tot+=len(x[i])
    x0.append(x[i][0][0])
tot

# %%
plt.hist(x0, bins=100, color='blue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('x0 joint 0 distribution')
 
# Display the plot
plt.show()

# %%
x0resampled=np.load('/home/utente/Documents/Optim/mpc-dock-default2/safe-mpc/data/x_init_2.npy')

# %%
plt.hist(x0resampled[:,0], bins=20, color='blue', edgecolor='black')
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('x0 joint 0 distribution')
 
# Display the plot
plt.show()

# %%
episode_len=[]
for i in range(len(data['safehor_hist'])):
    episode_len.append(len(data['safehor_hist'][i]))

plt.figure()
plt.title('Steps for episode')
plt.grid(True)
plt.plot(episode_len)
plt.show()
plt.close()

# %%
x = data['x_u']
u = data['x_u']

# %%
folder_name = os.getcwd()+'/plots29-07'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
folder_name

# %%
nplot = 0
for i in range(0,100):    

    x_plt=np.array(x['x_sim'][nplot])
    u_plt=np.array(u['u_sim'][nplot])

    plt.figure(figsize=(10, 10))
    plt.title('HPIPM')
    plt.subplot(1, 2, 1)    
    plt.grid(True)
    plt.plot(np.arange(0,len(x_plt[:,0]))*5e-3,x_plt[:,0])
    plt.subplot(1, 2, 2)    
    plt.grid(True)
    plt.plot(np.arange(0,len(x_plt[:,3]))*5e-3,x_plt[:,3])
    plt.savefig(os.path.join(folder_name, str(nplot)+'hpipm'+'.png'))
    plt.close()


    plt.figure()
    plt.title('HPIPM u')
    plt.grid(True)
    plt.plot(np.arange(0,len(u_plt[:,0]))*5e-3,u_plt[:,0],label='u1')
    plt.plot(np.arange(0,len(u_plt[:,0]))*5e-3,u_plt[:,1],label='u2')
    plt.plot(np.arange(0,len(u_plt[:,0]))*5e-3,u_plt[:,2],label='u3')
    plt.legend()
    plt.savefig(os.path.join(folder_name, 'u'+str(nplot)+'hpipm'+'.png'))
    plt.close()

    nplot +=1


