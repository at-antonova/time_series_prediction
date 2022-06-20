from predictor import TimeSeriesPredictor, RapidGrowthDBSCANNPM, LimitClusterSizeNPM, WeirdPatternsNPM
import pickle
import numpy as np


motif_clustering_params = {
    'beta': 0.2,
    'mc_method': 'db'
}

prediction_params = {
    'up_method' : 'db', # {'a', 'wi', 'db', 'op'}
    'eps' : 0.01, # max distance in cluster
    'min_samples' : 5, # min number of samples in cluster
    'cluster_size_threshold': 0, # Minimal percentage of points in largest cluster to call point predictable
    'one_cluster_rule': False, #Point is predictable only is there is one cluster (not including noise)
    'alg_type' : 's', # {'s', 'tp'}
    'np_method' : 'rd', # {'fp', 'ls', 'rg', 'rd', 'rw'}
}

healing_params = {
    'healing_up_method' : 'db', 
    'weighted_up' : True,  
    # {'double_clustering', 'weighred_average', 'pointwise_weights', 
    # 'pattern_length', 'pattern_length_dist', 'dist', 
    # 'dist_factor'}
    'weight_method' : 'pointwise_weights', 
    'clear_noise' : False,
    'factor' : 0.9,
    'alg_type' : 's',
    'mc_method' : 'db',
    'beta' : 0.2,
    'fixed_points' : False,
    'healing_motif_match_eps': 0.01
}

ROOT_PATH = '/Users/nastya/dev/time_series_prediction/'

# prediction horizon
h = 100

# load Lorenz time series
n_train = 10_000
n_test = 1_000 + 300 # to get test set of 1000
n_passed = 3_000
n_valid = 2_000

with open(ROOT_PATH + 'data/lorenz.dat', 'rb') as f:
    Y = pickle.load(f)

Y1 = np.array(Y[n_passed:n_passed + n_train]).reshape(-1)
Y2 = np.array(Y[n_passed + n_train:n_passed + n_train + n_test]).reshape(-1)
Y3 = np.array(Y[n_passed + n_train + n_test:n_passed + n_train + n_test + n_valid]).reshape(-1)


try:
    mc_method = motif_clustering_params.get('mc_method', 'db')
    beta_str = str(int(motif_clustering_params.get('beta', 0.2) * 100))
    motifs_filename = 'motifs_' + mc_method + '_b' + beta_str + '.dat'
    with open(ROOT_PATH + 'data/motifs/' + motifs_filename, 'rb') as f:
        clustered_motifs = pickle.load(f)
except FileNotFoundError:
    print('File with saved motifs not found!')


base_npm = LimitClusterSizeNPM(0.5, 2)
predictor1 = TimeSeriesPredictor(clustered_motifs, base_npm, 10, 3)
        
Y_preceding = Y1[-31:]
unified_preds, possible_predictions_list = \
    predictor1.predict(Y_preceding, h, **prediction_params)

base_npm = LimitClusterSizeNPM(0.5, 2)
npm = WeirdPatternsNPM(Y1, eps0=0.07, base_non_pred_model=base_npm)
predictor1.set_non_pred_model(npm)

up, pp, n_iterations = predictor1.self_healing(Y_preceding, h, return_n_iterations=True, \
    unified_predictions=unified_preds, \
    possible_predictions=possible_predictions_list, \
    healing_logs_filepath='logs_ba_db_db_wfactor09_weirdpateps007_rd_lcs_01_20.dat',
    fixed_points_idx=[],
    **healing_params)