# Import packages 
import numpy as np
import pynbody as pb
import pickle 

cptmarvel_path = f"/myhome2/users/munshi/dwarf_volumes/cptmarvel.cosmo25cmb.4096g5HbwK1BH/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096/cptmarvel.cosmo25cmb.4096g5HbwK1BH.004096"
        #halos : [1,2,3,5,6,7,10,11,13,14,24]
elektra_path = f"/myhome2/users/munshi/dwarf_volumes/elektra.cosmo25cmb.4096g5HbwK1BH/elektra.cosmo25cmb.4096g5HbwK1BH.004096/elektra.cosmo25cmb.4096g5HbwK1BH.004096"
        #halos : [1,2,3,4,5,8,9,10,11,12,17,36,64]
storm_path = f"/myhome2/users/munshi/dwarf_volumes/storm.cosmo25cmb.4096g5HbwK1BH/storm.cosmo25cmb.4096g5HbwK1BH.004096/storm.cosmo25cmb.4096g5HbwK1BH.004096"
        #halos : [1,2,3,4,5,6,7,8,10,11,12,14,15,22,23,31,37,44,48,55,118]
rogue_path = f"/myhome2/users/munshi/dwarf_volumes/rogue.cosmo25cmb.4096g5HbwK1BH/rogue.cosmo25cmb.4096g5HbwK1BH.004096/rogue.cosmo25cmb.4096g5HbwK1BH.004096"
        #halos: [1,3,7,8,10,11,12,15,16,17,28,31,37,58,116] 
#storm_bubbles = f"/myhome2/users/munshi/dwarf_volumes/storm.cosmo25cmb.4096g1HsbBH.004096/storm.cosmo25cmb.4096g1HsbBH.004096"
        #halos: [1,2,3,4,5,6,7,8,10,11,12,14,15,18,21,23,35,42,48,49,61,88,125,133,175,186,235,262,272,300,541] 
#storm_bubbles = f"/myhome2/users/munshi/dwarf_volumes/storm.cosmo25cmb.4096g1HsbBH.004096/storm.cosmo25cmb.4096g1HsbBH.004096"
storm_bubbles = f"/myhome2/users/munshi/dwarf_volumes/storm.cosmo25cmb.4096g1HsbBH/storm.cosmo25cmb.4096g1HsbBH.004096"

# Loading halo
print("Loading halo of interest...")
s = pb.load(storm_bubbles) #load path 
s.physical_units()
#h = s.halos() #select for blastwave
h = s.halos(dosort=True) #select for superbubbles
#halos = [1,2,3,5,6,7,10,11,13,14,24] #select for Cpt Marvel
#halos = [1,2,3,4,5,8,9,10,11,12,17,36,64] #select for Elektra
#halos = [1,2,3,4,5,6,7,8,10,11,12,14,15,22,23,37,44,48,55,118] #select for Storm
#halos = [1,2,3,4,5,6,7,8,10,11,12,14,15,22,23,37,44,48,55,118] #storm2
#halos = [1,3,7,8,10,11,12,15,16,17,28,31,37,58,116] #select for Rogue
#halos = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 19, 21, 23, 25, 41, 42, 61, 82]#select for Storm SuperBubbles
halos = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 20, 22, 23, 24, 36, 48, 76] #SuperBubble2
###--------------------------------------sntiming.py//-------------------------------------------###

def expbins(simbefore, dt=1.e6, tstart=0, tmax=13.8e9, sniimax=40.0):
    """Gives the number of supernova explosions in each bin for non-stochastic IMF sim

    If tstart>0, shifts the initial time (e.g., the t=0 time) by that much, in years
    
    tmax - the maximum time to consider (relative to tstart)
    
    returns bins in Gyr and number of SN (divide by bin size to make into rate)

    *** for example ***
        tstart=0.5e9, tmax=2.e9, dt=1.e6
        
        This will return the supernova rate between 0.5 Gyr and 2.5 Gyr relative
        to the beginning of the simulation
        
        However, the bins will be returned as 0 to 2 Gyr in 1 Myr increments
    """
    bins=np.arange(0, tmax+dt, dt)
    alltimes = np.array([])
    allnums = np.array([])
    notbhs = pb.filt.HighPass('tform', '0 Gyr')
    sim = simbefore.s[notbhs]
    ts = np.arange(0, 50.e6+dt, dt)
    for t in np.arange(0, 50.e6+dt, dt):
        mmax = slt.starmass(t, sim['metals'])
        mmax[mmax>sniimax] = sniimax
        mmin = slt.starmass(t+dt, sim['metals'])
        mmin[mmin<8.] = 8.
        nums = (imf.CumNumber(mmin) - imf.CumNumber(mmax))*sim['massform'].in_units('Msol')
        nums[mmax<8.] = 0.
        nums[mmin>sniimax] = 0.
        times = np.array(sim['tform'].in_units('yr')) + t + dt
        alltimes = np.concatenate((alltimes, times))
        allnums = np.concatenate((allnums, nums))

    alltimes = np.array(alltimes)
    allnums = np.array(allnums)

    if tstart>0:
        alltimes -= tstart

    vals = []
    for i in range(len(bins)-1):
        vals.append(np.sum(allnums[(alltimes>=bins[i]) & (alltimes<bins[i+1])]))
    
    vals, bins =  np.array(vals), np.array(bins)
    return vals, bins/1.e9
      
###-------------------------------------------Burstiness-------------------------------------------------------###

def get_burstiness(data,time_bins,bin_size):
    burstiness = []
    burst_time = []
    for i in np.arange(len(data)-bin_size):
        sub_data = data[i:i+bin_size]
        time = time_bins[i:i+bin_size]
        burst_time.append(np.mean(time))
        mu = np.mean(sub_data)
        sigma = np.std(sub_data)
        burstiness.append((sigma/mu-1)/(sigma/mu+1) )
    return burstiness, burst_time
   
Data = {}
# Calculations
print('Calcualting SNR and Burstiness...')
for hnum in halos:
    snr,tbins = expbins(h[hnum])
    print('halo' + str(hnum) + 'snr done')
    burstiness,t_burst = get_burstiness(snr,tbins,50)
    print('halo' + str(hnum) + 'burstiness done')
    new_burstiness = np.nan_to_num(burstiness,nan=-1)
    print('halo' + str(hnum) + 'new burstiness done')
    Data[str(hnum)] = {}
    halo_dir_path = "/myhome2/users/azartash/sncalc/storm_bubbles_halos" #pathway to where you want output files saved to
    os.chdir(halo_dir_path)
    Data[str(hnum)]['SNR'] = snr
    Data[str(hnum)]['SN_bins'] = tbins
    Data[str(hnum)]['Burstiness'] = new_burstiness
    Data[str(hnum)]['Burstiness_bins'] = t_burst
    print('halo' + str(hnum) + 'done')
out = open('storm_bubbles_burstiness.pickle', 'wb')
pickle.dump(Data,out)
out.close
