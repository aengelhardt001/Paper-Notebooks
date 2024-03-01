# Import packages 
import numpy as np
import pynbody as pb
import matplotlib.pyplot as plt
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
storm_bubbles = f"/myhome2/users/munshi/dwarf_volumes/storm.cosmo25cmb.4096g1HsbBH/storm.cosmo25cmb.4096g1HsbBH.004096"

# Loading halo
print("Loading halo of interest...")
s = pb.load(storm_bubbles) # load marvel_path 
s.physical_units()
#h = s.halos() # use for blastwave 
h = s.halos(DoSort=True) # use for superbubble
#halos = [1,2,3,5,6,7,10,11,13,14,24] # cpt marvel
#halos = [1,2,3,4,5,8,9,10,11,12,17,36,64] # elektra
#halos = [1,2,3,4,5,6,7,8,10,11,12,14,15,22,23,31,37,44,48,55,118] # storm
#halos = [1,2,3,4,5,6,7,8,10,11,12,14,15,22,23,37,44,48,55,118] # storm2
#halos = [1,3,7,8,10,11,12,15,16,17,28,31,37,58,116] # rogue
#halos = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 19, 21, 23, 25, 41, 42, 61, 82] # storm superbubbles 
halos = halos = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 20, 22, 23, 24, 36, 48, 76] # superbubble2

halo_dir_path = "/myhome2/users/azartash/sncalc/storm_bubbles_halos" #pathway to output files//where figures will be saved
os.chdir(halo_dir_path)
 

def last_star_formation(halo):
    tform = halo.star['tform'].in_units("Gyr")
    return np.max(tform)

   
def get_stellar_mass(halo):
    stellar_mass = halo.star['mass'].in_units("Msol").sum()
    return stellar_mass


def get_virial_mass(halo):
    virial_mass = halo['mass'].sum()
    return virial_mass

Data = {}

# Calculations
print('Calcualting last star formation, virial mass, stellar mass')

for hnum in halos:
    star_form_bins = last_star_formation(h[hnum]) 
    M_star = get_stellar_mass(h[hnum])
    M_vir = get_virial_mass(h[hnum])
    M_halo = M_star/M_vir
    Data[str(hnum)] = {}
    Data[str(hnum)]['Last Star'] = star_form_bins
    Data[str(hnum)]['Stellar Mass'] = M_star
    Data[str(hnum)]['Virial Mass'] = M_vir
    Data[str(hnum)]['Halo Mass'] = M_halo
out = open('storm_bubbles_avg_burstiness.pickle', 'wb') #change to match SIM
pickle.dump(Data,out)
out.close

