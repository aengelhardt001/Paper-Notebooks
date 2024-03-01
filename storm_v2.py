import numpy as np
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

from sklearn.linear_model import LinearRegression


sncalc_dir = '/myhome2/users/azartash/sncalc'

cptm_halo_dir_path = f'{sncalc_dir}/cptmarvel_halos'
elektra_halo_dir_path = f'{sncalc_dir}/elektra_halos'
storm_halo_dir_path = f'{sncalc_dir}/storm_halos'
rogue_halo_dir_path = f'{sncalc_dir}/rogue_halos'
storm_bubbles_path = f'{sncalc_dir}/storm_bubbles_halos'

bw_slopes = np.loadtxt('BW_slopes.txt')
sb_slopes = np.loadtxt('SB_slopes.txt')

fn = f'{storm_bubbles_path}/storm_bubbles_burstiness.pickle'
sb_data = pickle.load(open(fn, 'rb'))
fn = f'{storm_bubbles_path}/storm_bubbles_avg_burstiness.pickle'
sb_Data = pickle.load(open(fn, 'rb'))

sb_burstiness, sb_avg_burstiness, sb_stellar_mass, \
    sb_virial_mass, sb_avg_active_burstiness, sb_inst_burstiness = \
    [], [], [], [], [], []

# sb_halos = [
#     1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 19, 21, 23, 25, 41, 42, 61, 82
# ]
sb_halos = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 20, 22, 23, 24, 36, 48, 76
]

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


fn = f'{storm_halo_dir_path}/storm_burstiness.pickle'
s_data = pickle.load(open(fn, 'rb'))
fn = f'{storm_halo_dir_path}/storm_avg_burstiness.pickle'
s_Data = pickle.load(open(fn, 'rb'))

s_burstiness, s_avg_burstiness, s_stellar_mass, \
    s_virial_mass, s_avg_active_burstiness, s_inst_burstiness = \
    [], [], [], [], [], []

s_halos = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 22, 23, 37, 44, 48, 55, 118
]

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


bw_halo_mass = [i / j for i, j in zip(s_stellar_mass, s_virial_mass)]
sb_halo_mass = [i / j for i, j in zip(sb_stellar_mass, sb_virial_mass)]

s_avg_active_burstiness = np.array(s_avg_active_burstiness)
sb_avg_active_burstiness = np.array(sb_avg_active_burstiness)
s_avg_burstiness = np.array(s_avg_burstiness)
sb_avg_burstiness = np.array(sb_avg_burstiness)
bw_slopes = np.array(bw_slopes)
sb_slopes = np.array(sb_slopes)
bw_halo_mass = np.array(bw_halo_mass)
sb_halo_mass = np.array(sb_halo_mass)


# SETUP PLOT
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.major.top'] = True
mpl.rcParams['xtick.major.top'] = True
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.major.right'] = True
mpl.rcParams['ytick.minor.right'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8.0, 6.0))

ax_pri_0_0 = ax[0, 0]
ax_pri_0_1 = ax[0, 1]
ax_pri_1_0 = ax[1, 0]
ax_pri_1_1 = ax[1, 1]

ax_sec_0_0 = ax[0, 0].twinx()
ax_sec_0_1 = ax[0, 1].twinx()
ax_sec_1_0 = ax[1, 0].twinx()
ax_sec_1_1 = ax[1, 1].twinx()


def truncate_colormap(cmap, min_val=0., max_val=1., n=100):
    '''Custom colormap'''
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name}, {min_val:.2f}, {max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n))
    )
    return new_cmap


# cmap = plt.get_cmap('viridis')
cmap = LinearSegmentedColormap.from_list('', ['#E41A1C', 'violet', '#377EB8'])
# cmap = truncate_colormap(cmap, 0.1, 0.85)

burst_params = {
    's': 24.,
    'cmap': cmap,
    'norm': Normalize(4, 9),
    # 'edgecolors': 'k'
}
alpha_params = {
    'c': 'k',
    'marker': '+',
}
burst_line_params = {
    'c': 'r',
    'alpha': 0.7,
    'zorder': -10,
}
alpha_line_params = {
    'c': 'k',
    'ls': '--',
    'alpha': 0.7,
    'zorder': -10,
}


def plot_panel(ax_pri, ax_sec, x, y_pri, y_sec, c_pri, label):
    sc_pri = ax_pri.scatter(x, y_pri, c=c_pri, label=f'Burstiness',
                            **burst_params)
    sc_sec = ax_sec.scatter(x, y_sec, label=rf'$\alpha$',
                            **alpha_params)

    # LINEAR REGRESSIONS
    log_x = np.log10(x)

    pri_model = LinearRegression().fit(log_x.reshape(-1, 1),
                                       y_pri.reshape(-1, 1))
    sec_model = LinearRegression().fit(log_x.reshape(-1, 1),
                                       y_sec.reshape(-1, 1))

    pri_int, pri_slope = pri_model.intercept_[0], pri_model.coef_[0][0]
    sec_int, sec_slope = sec_model.intercept_[0], sec_model.coef_[0][0]

    pri_R2 = pri_model.score(log_x.reshape(-1, 1), y_pri.reshape(-1, 1))
    sec_R2 = sec_model.score(log_x.reshape(-1, 1), y_sec.reshape(-1, 1))
    # print(pri_R2, pri_R2)

    line_x = np.array([x.min(), x.max()])
    line_y_pri = pri_int + np.log10(line_x) * pri_slope
    line_y_sec = sec_int + np.log10(line_x) * sec_slope

    # label5 = f'intercept = {pri_int:.3f}\nslope = {pri_slope:.3f}\n$R^2$ = {pri_R2:.3f}'
    # label6 = f'intercept = {sec_int:.3f}\nslope = {sec_slope:.3f}\n$R^2$ = {sec_R2:.3f}'
    label_pri = f'y-int. = {pri_int:.3f}\nslope = {pri_slope:.3f}\n$R^2$ = {pri_R2:.3f}'
    label_sec = f'y-int. = {sec_int:.3f}\nslope = {sec_slope:.3f}\n$R^2$ = {sec_R2:.3f}'

    sc_line_pri = ax_pri.plot(line_x, line_y_pri, label=label_pri,
                              **burst_line_params)[0]
    sc_line_sec = ax_sec.plot(line_x, line_y_sec, label=label_sec,
                              **alpha_line_params)[0]

    # LABEL
    ax_pri.text(0.94, 0.12, label, transform=ax_pri.transAxes, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='k', pad=3.5), zorder=10)

    # LEGEND
    sc_list = [sc_pri, sc_sec, sc_line_pri, sc_line_sec]
    labels = [sc.get_label() for sc in sc_list]

    ax_pri.legend(sc_list, labels,
                  fontsize=6.5, frameon=False, loc='upper left')

    return sc_pri, sc_sec, sc_line_pri, sc_line_sec


sc_burst_bw_avg, sc_alpha_bw_avg, sc_line_burst_bw_avg, sc_line_alpha_bw_avg = \
    plot_panel(ax_pri_0_0, ax_sec_0_0,
               bw_halo_mass, s_avg_burstiness, bw_slopes,
               np.log10(s_stellar_mass), 'BW, Mean')

sc_burst_sb_avg, sc_alpha_sb_avg, sc_line_burst_sb_avg, sc_line_alpha_sb_avg = \
    plot_panel(ax_pri_0_1, ax_sec_0_1,
               sb_halo_mass, sb_avg_burstiness, sb_slopes,
               np.log10(sb_stellar_mass), 'SB, Mean')

sc_burst_bw_active, sc_alpha_bw_active, sc_line_burst_bw_active, sc_line_alpha_bw_active = \
    plot_panel(ax_pri_1_0, ax_sec_1_0,
               bw_halo_mass, s_avg_active_burstiness, bw_slopes,
               np.log10(s_stellar_mass), 'BW, Active')

sc_burst_sb_active, sc_alpha_sb_active, sc_line_burst_sb_active, sc_line_alpha_sb_active = \
    plot_panel(ax_pri_1_1, ax_sec_1_1,
               sb_halo_mass, sb_avg_active_burstiness, sb_slopes,
               np.log10(sb_stellar_mass), 'SB, Active')


ax_pri_0_0.set_xscale('log')


# LABELS
fig.text(0.5, 0.04, r'M$_\star$ / M$_\mathrm{halo}$',
         va='center', ha='center', fontsize=15)
ax_pri_0_0.set_ylabel('Mean Burstiness', fontsize=15.)
ax_pri_1_0.set_ylabel('Mean Active Burstiness', fontsize=15.)
ax_sec_0_1.set_ylabel(r'$\alpha$', fontsize=15)
ax_sec_1_1.set_ylabel(r'$\alpha$', fontsize=15)


# CBAR
cbar_ax = fig.add_axes([0.92, 0.13, 0.03, 0.82])
cbar = fig.colorbar(sc_burst_bw_avg, cax=cbar_ax)
cbar.set_label(r'M$_\star$ [M$_\odot$]', fontsize=15, color='black')


# TICKS
def set_major_minor(ax, axis, major_ticks, minor_ticks):
    if axis == 'x':
        ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))
    elif axis == 'y':
        ax.yaxis.set_major_locator(MultipleLocator(major_ticks))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_ticks))


[set_major_minor(_ax, 'y', 0.1, 0.025)
 for _ax in ax.ravel()]

set_major_minor(ax_sec_0_0, 'y', 0.2, 0.05)
set_major_minor(ax_sec_0_1, 'y', 0.2, 0.05)
set_major_minor(ax_sec_1_0, 'y', 0.2, 0.05)
set_major_minor(ax_sec_1_1, 'y', 0.2, 0.05)

[_ax.tick_params(axis='both', which='major', labelsize=12., length=5.)
 for _ax in ax.ravel()]
[_ax.tick_params(axis='both', which='minor', length=3.)
 for _ax in ax.ravel()]

ax_sec_0_0.tick_params(axis='both', which='minor', length=3.)
ax_sec_0_1.tick_params(axis='both', which='minor', length=3.)
ax_sec_1_0.tick_params(axis='both', which='minor', length=3.)
ax_sec_1_1.tick_params(axis='both', which='minor', length=3.)

ax_sec_0_0.yaxis.set_ticks([])
ax_sec_1_0.yaxis.set_ticks([])


# LIMIT SCALING
y_min = min(s_avg_burstiness.min(), sb_avg_burstiness.min(),
            s_avg_active_burstiness.min(), sb_avg_active_burstiness.min())
y_max = max(s_avg_burstiness.max(), sb_avg_burstiness.max(),
            s_avg_active_burstiness.max(), sb_avg_active_burstiness.max())
spacing = abs(y_max - y_min) * 0.08
ax_pri_0_0.set_ylim(y_min - spacing, spacing + y_max)

y_min = min(bw_slopes.min(), sb_slopes.min())
y_max = max(bw_slopes.max(), sb_slopes.max())
spacing = abs(y_max - y_min) * 0.08
ax_sec_0_0.set_ylim(y_min - spacing, spacing + y_max)
ax_sec_0_1.set_ylim(y_min - spacing, spacing + y_max)
ax_sec_1_0.set_ylim(y_min - spacing, spacing + y_max)
ax_sec_1_1.set_ylim(y_min - spacing, spacing + y_max)

# ADJUST & SAVE
fig.subplots_adjust(top=0.95, left=0.1, right=0.83, bottom=0.12, wspace=0.06, hspace=0.06)


plt.savefig('burst.pdf')
