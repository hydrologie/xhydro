import numpy as np
import os
import scipy.io
def apply_regional_weights_avg_climex():
    work_folder = os.getcwd()
    data_folder = 'C:/Users/AR21010/Dropbox/xhydro/INFO-Crue/0_data/2_extracted_data'
    results_folder = 'C:/Users/AR21010/Dropbox/xhydro/INFO-Crue/0_data/3_results'
    save_folder = 'C:/Users/AR21010/Dropbox/xhydro/INFO-Crue/0_data/3_results'

    # Type of simulations to use
    simulation_type = 'HYDROTEL_6'  # 6 HYDROTEL simulation with ClimEx
    # simulation_type = 'HYDROTEL_144';  # 144 HYDROTEL simulation with ClimEx
    # simulation_type = 'HYDROTEL_150';  # 150 HYDROTEL simulation with ClimEx

    # Take all observed watersheds or only a selection of them (made by DEH)
    obs_stations_to_select = 'selection'

    # Load data
    mat_obs_data = scipy.io.loadmat(
        os.path.join(data_folder, f'INFO_Crue_obs_stations_{simulation_type}_ClimEx_{obs_stations_to_select}.mat'))
    obs_data = mat_obs_data['all_Qsim']
    mat_avg_weights = scipy.io.loadmat(
        os.path.join(results_folder, f'AVG_{simulation_type}_{obs_stations_to_select}_weights.mat'))
    regional_weights = mat_avg_weights['regional_weights']
    mat_avg_data = scipy.io.loadmat(
        os.path.join(save_folder, f'AVG_{simulation_type}_{obs_stations_to_select}_ClimEx.mat'))
    all_data = mat_avg_data['all_data']

    # There should be 259 watersheds (all) or 96 watersheds (selection)
    n_watersheds = obs_data.shape[0]
    # Number of ClimEx members
    n_members = 50

    # All ClimEx 50-member simulation names
    ClimEx_sims = [
        'KDA', 'KDB', 'KDC', 'KDD', 'KDE', 'KDF', 'KDG', 'KDH', 'KDI', 'KDJ',
        'KDK', 'KDL', 'KDM', 'KDN', 'KDO', 'KDP', 'KDQ', 'KDR', 'KDS', 'KDT',
        'KDU', 'KDV', 'KDW', 'KDX', 'KDY', 'KDZ', 'KEA', 'KEB', 'KEC', 'KED',
        'KEE', 'KEF', 'KEG', 'KEH', 'KEI', 'KEJ', 'KEK', 'KEL', 'KEM', 'KEN',
        'KEO', 'KEP', 'KEQ', 'KER', 'KES', 'KET', 'KEU', 'KEV', 'KEW', 'KEX'
    ]

    # Combine all Qobs and Qsim together
    for i in range(n_watersheds):
        for j in range(n_members):
            # Extract all simulated streamflows for the current member
            Qsim = obs_data[i, 0][ClimEx_sims[j]]

            # Remove nans and February 29
            Qsim = np.delete(Qsim, np.s_[54786:], axis=0)  # December 30, 2099 to December 31, 2100
            Qsim = np.delete(Qsim, np.s_[:1826], axis=0)  # January 01, 1950 to December 31, 1954
            ind29feb = np.where((all_data[i, 0]['dates'][:, 1] == 2) & (all_data[i, 0]['dates'][:, 2] == 29))[0]
            Qsim = np.delete(Qsim, ind29feb, axis=0)  # All February 29th

            # Copy the last day twice since there are missing 2 days at the end
            # of the dataset (it ends at December 29). This will not affect the
            # results since only the annual maxima will be used afterwards.
            Qsim = np.vstack([Qsim, [Qsim[-1]] * 2])

            # Compute the AVG average
            Qsim = np.sum(Qsim * regional_weights, axis=1)

            # Store the averaged data into the structure
            all_data[i, 0]['Qsim'][:, j] = Qsim

    # Save the AVG results
    save_path = os.path.join(save_folder, f'AVG_{simulation_type}_{obs_stations_to_select}_ClimEx.mat')
    scipy.io.savemat(save_path, {'regional_weights': regional_weights, 'all_data': all_data}, format='5',
                     long_field_names=True)
