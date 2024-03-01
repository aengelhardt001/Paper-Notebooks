import numpy as np
import pynbody as pb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from colorspacious import cspace_converter
from matplotlib import rc
import pickle
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from matplotlib.markers import MarkerStyle


cptm_halo_dir_path = "/myhome2/users/azartash/sncalc/cptmarvel_halos/"
elektra_halo_dir_path = "/myhome2/users/azartash/sncalc/elektra_halos/"
storm_halo_dir_path = "/myhome2/users/azartash/sncalc/storm_halos/"
rogue_halo_dir_path = "/myhome2/users/azartash/sncalc/rogue_halos/"
storm_bubbles_path = "/myhome2/users/azartash/sncalc/storm_bubbles_halos/"

bw_slope_data = np.loadtxt('BW_slopes.txt')
bw_slopes = np.array(bw_slope_data)
sb_slope_data = np.loadtxt('SB_slopes.txt')
sb_slopes = np.array(sb_slope_data)

sb_data = pickle.load(open(storm_bubbles_path +'storm_bubbles_burstiness.pickle', 'rb'))
sb_Data = pickle.load(open(storm_bubbles_path +'storm_bubbles_avg_burstiness.pickle', 'rb'))
sb_burstiness, sb_avg_burstiness , sb_stellar_mass, sb_virial_mass, sb_avg_active_burstiness, sb_inst_burstiness = [],[],[],[],[],[]
#sb_halos = [1,2,3,4,5,6,7,8,10,11,12,14,15,19,21,23,25,41,42,61,82]
sb_halos = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 20, 22, 23, 24, 36, 48, 76]
#for h in ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '14', '15', '19', '21', '23', '25', '41', '42', '61', '82']:
for halo_num in sb_halos:
    halo_data = sb_data[str(halo_num)]
    halo_Data = sb_Data[str(halo_num)]
    burst = np.array(halo_data['Burstiness'])
    sb_t_burst = np.array(halo_data['Burstiness_bins'])
    sb_last_star_form = halo_Data['Last Star']
    active_mask = sb_t_burst < sb_last_star_form
    burst_active = burst[active_mask]
    sb_t_burst = sb_t_burst[active_mask]
    sb_burstiness.append(burst)
    sb_stellar_mass.append(halo_Data['Stellar Mass'])
    sb_virial_mass.append(halo_Data['Virial Mass'])
    sb_avg_burstiness.append(burst.mean())
    sb_avg_active_burstiness.append(burst_active.mean())
    sb_inst_burstiness.append(burst_active[-2:].mean())

   


s_data = pickle.load(open(storm_halo_dir_path +'storm_burstiness.pickle', 'rb'))
s_Data = pickle.load(open(storm_halo_dir_path +'storm_avg_burstiness.pickle', 'rb'))
s_burstiness, s_avg_burstiness , s_stellar_mass, s_virial_mass, s_avg_active_burstiness, s_inst_burstiness = [],[],[],[],[],[]
s_halos = [1,2,3,4,5,6,7,8,10,11,12,14,15,22,23,37,44,48,55,118]
#for h in ['1','2','3','4','5','6','7','8','10','11','12','14','15','22','23','37','44','48','55','118']:
for halo_num in s_halos:
    halo_data = s_data[str(halo_num)]
    halo_Data = s_Data[str(halo_num)]
    burst = np.array(halo_data['Burstiness'])
    s_t_burst = np.array(halo_data['Burstiness_bins'])
    s_last_star_form = halo_Data['Last Star']
    active_mask = s_t_burst < s_last_star_form
    burst_active = burst[active_mask]
    s_t_burst = s_t_burst[active_mask]
    s_burstiness.append(burst)
    s_stellar_mass.append(halo_Data['Stellar Mass'])
    s_virial_mass.append(halo_Data['Virial Mass'])
    s_avg_burstiness.append(burst.mean())
    s_avg_active_burstiness.append(burst_active.mean())
    s_inst_burstiness.append(burst_active[-2:].mean())

 
  
bw_halo_mass = [i/j for i, j in zip(s_stellar_mass, s_virial_mass)]
sb_halo_mass = [i/j for i, j in zip(sb_stellar_mass, sb_virial_mass)]

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.major.top'] = True
mpl.rcParams['xtick.major.top'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.major.right'] = True
mpl.rcParams['ytick.minor.right'] = True

plt.rc('text',usetex=True)
plt.rc('font',family='serif')
#fig, ax = plt.subplots()
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True,figsize=(9.5,4.8))
sec_ax_left = ax[0].twinx()
sec_ax_right = ax[1].twinx()
lower_bound, upper_bound = 4,9
norm = Normalize(lower_bound,upper_bound)
# s1 = (np.array(bw_slopes)+2)**3*7
# s2 = (np.array(sb_slopes)+2)**3*7
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
#cmap = plt.get_cmap('viridis')
cmap = mpl.colors.LinearSegmentedColormap.from_list('',['#E41A1C','violet','#377EB8'])
# new_cmap = truncate_colormap(cmap, 0.1, 0.85)
plot_params = {
    's': 30.,
    'cmap': cmap,
    'norm': norm,
    # 'edgecolors': 'k'
}
sc1 = ax[0].scatter(bw_halo_mass, s_avg_active_burstiness, c=np.log10(s_stellar_mass), label='Burstiness', **plot_params)
sc2 = sec_ax_left.scatter(bw_halo_mass, bw_slopes, color="black", marker="+", label=r'BW $\alpha$')
sc3 = ax[1].scatter(sb_halo_mass, sb_avg_active_burstiness, c=np.log10(sb_stellar_mass), label='Burstiness', **plot_params)
sc4 = sec_ax_right.scatter(sb_halo_mass, sb_slopes, color="black", marker="+", label=r'SB $\alpha$')


ax[1].semilogx()
fig.text(0.5, 0.04, r'M$_\star$ / M$_\mathrm{halo}$', va='center', ha='center', fontsize=15)
ax[0].set_ylabel('Mean Active Burstiness', fontsize=15.)
sec_ax_right.set_ylabel(r'$\alpha$', fontsize=15)
fig.subplots_adjust(top=0.95, left=0.07, right=0.85, bottom=0.13, wspace=0.05)

cbar_ax = fig.add_axes([0.92,0.13,0.03,0.82])
cbar2 = fig.colorbar(sc1, cax=cbar_ax)
cbar2.set_label(r'M$_\star$ [M$_\odot$]', fontsize=15,  color='black')

bw_y = np.array(s_avg_active_burstiness)
sb_y = np.array(sb_avg_active_burstiness)
bw_y_alpha = np.array(bw_slopes)
sb_y_alpha = np.array(sb_slopes)
bw_x = np.log10(bw_halo_mass)
sb_x = np.log10(sb_halo_mass)


# x and y are the data points

bw_burst_model = LinearRegression().fit(bw_x.reshape(-1, 1), bw_y.reshape(-1, 1))
bw_alpha_model = LinearRegression().fit(bw_x.reshape(-1, 1), bw_y_alpha.reshape(-1, 1))
sb_burst_model = LinearRegression().fit(sb_x.reshape(-1, 1), sb_y.reshape(-1, 1))
sb_alpha_model = LinearRegression().fit(sb_x.reshape(-1, 1), sb_y_alpha.reshape(-1,1))

bw_burst_y_int, bw_burst_slope = bw_burst_model.intercept_[0], bw_burst_model.coef_[0][0]
bw_alpha_y_int, bw_alpha_slope = bw_alpha_model.intercept_[0], bw_alpha_model.coef_[0][0]
sb_burst_y_int, sb_burst_slope = sb_burst_model.intercept_[0], sb_burst_model.coef_[0][0] 
sb_alpha_y_int, sb_alpha_slope = sb_alpha_model.intercept_[0], sb_alpha_model.coef_[0][0] 

bw_burst_R2 = bw_burst_model.score(bw_x.reshape(-1, 1), bw_y.reshape(-1, 1))
bw_alpha_R2 = bw_alpha_model.score(bw_x.reshape(-1, 1), bw_y_alpha.reshape(-1, 1))
sb_burst_R2 = sb_burst_model.score(sb_x.reshape(-1, 1), sb_y.reshape(-1, 1))
sb_alpha_R2 = sb_alpha_model.score(sb_x.reshape(-1,1), sb_y_alpha.reshape(-1,1)) 

bw_burst_x_line = np.array([np.min(bw_halo_mass), np.max(bw_halo_mass)])   # Define how far you want the line to go
bw_burst_y_line = bw_burst_y_int + np.log10(bw_burst_x_line) * bw_burst_slope
bw_alpha_x_line = np.array([np.min(bw_halo_mass), np.max(bw_halo_mass)])   # Define how far you want the line to go
bw_alpha_y_line = bw_alpha_y_int + np.log10(bw_alpha_x_line) * bw_alpha_slope
sb_burst_x_line = np.array([np.min(sb_halo_mass), np.max(sb_halo_mass)])   # Define how far you want the line to go
sb_burst_y_line = sb_burst_y_int + np.log10(sb_burst_x_line) * sb_burst_slope
sb_alpha_x_line = np.array([np.min(sb_halo_mass), np.max(sb_halo_mass)])   # Define how far you want the line to go
sb_alpha_y_line = sb_alpha_y_int + np.log10(sb_alpha_x_line) * sb_alpha_slope



label5 = f'intercept = {bw_burst_y_int:.3f}\nslope = {bw_burst_slope:.3f}\n$R^2$ = {bw_burst_R2:.3f}'
label6 = f'intercept = {bw_alpha_y_int:.3f}\nslope = {bw_alpha_slope:.3f}\n$R^2$ = {bw_alpha_R2:.3f}'
label7 = f'intercept = {sb_burst_y_int:.3f}\nslope = {sb_burst_slope:.3f}\n$R^2$ = {sb_burst_R2:.3f}'
label8 = f'intercept = {sb_alpha_y_int:.3f}\nslope = {sb_alpha_slope:.3f}\n$R^2$ = {sb_alpha_R2:.3f}'

sc5 = ax[0].plot(bw_burst_x_line, bw_burst_y_line, color='red', zorder=-10, alpha=0.7,
    label=label5)[0]
sc6 = sec_ax_left.plot(bw_alpha_x_line, bw_alpha_y_line, color='black', linestyle='dashed', zorder=-10, alpha=0.7,
    label=label6)[0]
sc7 = ax[1].plot(sb_burst_x_line, sb_burst_y_line, color='red', zorder=-10, alpha=0.7,
    label=label7)[0]
sc8 = sec_ax_right.plot(sb_alpha_x_line, sb_alpha_y_line, color='black', linestyle='dashed', zorder=-10, alpha=0.7,
    label=label8)[0]


ax[0].yaxis.set_major_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.025))
sec_ax_left.yaxis.set_major_locator(MultipleLocator(0.2))
sec_ax_left.yaxis.set_minor_locator(MultipleLocator(0.05))
sec_ax_right.yaxis.set_major_locator(MultipleLocator(0.2))
sec_ax_right.yaxis.set_minor_locator(MultipleLocator(0.05))

[_ax.tick_params(axis='both', which='major', labelsize=12., length=5.) for _ax in ax]
[_ax.tick_params(axis='both', which='minor', length=3.) for _ax in ax]

sec_ax_left.tick_params(axis='both', which='minor', length=3.)
sec_ax_right.tick_params(axis='both', which='minor', length=3.)

y_min = min(np.min(s_avg_active_burstiness), np.min(sb_avg_active_burstiness)) 
y_max = max(np.max(s_avg_active_burstiness), np.max(sb_avg_active_burstiness))
spacing = abs(y_max - y_min) * 0.05
ax[0].set_ylim(y_min - spacing, spacing + y_max)

y_min = min(np.min(bw_slopes), np.min(sb_slopes)) 
y_max = max(np.max(bw_slopes), np.max(sb_slopes))
spacing = abs(y_max - y_min) * 0.05
sec_ax_left.set_ylim(y_min - spacing, spacing + y_max)
sec_ax_right.set_ylim(y_min - spacing, spacing + y_max)

labels = []
labels.append(sc1.get_label())
labels.append(sc2.get_label())
labels.append(sc5.get_label())
labels.append(sc6.get_label())
ax[0].legend([sc1, sc2, sc5, sc6], labels, fontsize=10., frameon=False, loc='upper left')

labels = []
labels.append(sc3.get_label())
labels.append(sc4.get_label())
labels.append(sc7.get_label())
labels.append(sc8.get_label())
ax[1].legend([sc3, sc4, sc7, sc8], labels, fontsize=10., frameon=False, loc='upper left')


sec_ax_left.yaxis.set_ticks([])



#plt.tight_layout()
plt.savefig('fit_avg_active_burst.pdf')
plt.show()
