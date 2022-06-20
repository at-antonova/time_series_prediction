import pickle
from predictor import TimeSeriesPredictor
from itertools import product
from joblib import Parallel, delayed
import numpy as np

# root path to working directory
ROOT_PATH = '/Users/nastya/dev/time_series_prediction/'

def experiment_no_pm(
    Y2, 
    h_max, 
    iterations_range, 
    motif_clustering_params, 
    prediction_params,
    healing_params, 
    non_pred_model_prediction,
    non_pred_model_healing,
    logs_filepath,
    n_test_passed=131, hs=[], n_iterations=-1):
    '''Main experiment, does not use prediction matrix, parallel execution
    
    Parameters
    ----------
    Y2: list or 1D np.ndarray
        Test set, the first 130 observations not counted in experiment
    h_max: int
        Max prediction horizon
    n_iterations: int
        Number of iterations or size of test set
    iterations_range: tuple
        Min and max n of iteration
    motif_clustering_params: dict
        'beta': percentage of used patterns
        'mc_method': method of clustering: 'db' for DBSCAN, 'wi' for Wishart
    prediction_params: dict
        Prediction paramenters
    healing_params: dict
        Healing parameters
    non_pred_model_prediction : NonPredModel
        NonPredModel for base algorithm prediction
    non_pred_model_healing : NonPredModel
        NonPredModel for self-healing algorithm
    '''
    
    def exp_task_no_pm(h, test_i):
        predictor1 = TimeSeriesPredictor(clustered_motifs, non_pred_model_prediction)
        Y_preceding = Y2[:test_i + n_test_passed - h + 1]
        unified_preds, possible_predictions_list = \
            predictor1.predict(Y_preceding, h, **prediction_params)
        predictor1.set_non_pred_model(non_pred_model_healing)
        up, pp = predictor1.self_healing(Y_preceding, h, return_n_iterations=False, \
            unified_predictions=unified_preds, \
            possible_predictions=possible_predictions_list, \
            healing_logs_filepath=None,
            fixed_points_idx=[],
            **healing_params)
        print(h, test_i)
        with open(logs_filepath, 'ab') as f:
            pickle.dump((h, test_i), f)
            pickle.dump(up[-1], f)


    # start experiment_no_pm    
    try:
        mc_method = motif_clustering_params.get('mc_method', 'db')
        beta_str = str(int(motif_clustering_params.get('beta', 0.2) * 100))
        motifs_filename = 'motifs_' + mc_method + '_b' + beta_str + '.dat'
        with open(ROOT_PATH + 'data/motifs/' + motifs_filename, 'rb') as f:
            clustered_motifs = pickle.load(f)
    except FileNotFoundError:
        print('File with saved motifs not found!')
    if h_max > 0:
        args = list(zip([h_max] * n_iterations, list(range(n_iterations)))) \
            + list(zip(list(range(1, h_max)), [n_iterations - 1] * h_max))
    else:
        args = list(product(hs, list(range(iterations_range[0], iterations_range[1]))))
        
    with open(logs_filepath, 'wb') as f:
        pass

    Parallel(n_jobs=-1, backend='threading')(delayed(exp_task_no_pm)(h0, i0) for (h0, i0) in args)


def experiment(
    Y2, 
    h_max, 
    n_iterations, 
    motif_clustering_params, 
    prediction_params,
    healing_params, 
    non_pred_model,
    logs_filepath,
    pm_filepath,
    n_test_passed=131, hs=[]):
    '''Function to perform an experiment with test set

    Parameters
    ----------
    Y2: list or 1D np.ndarray
        Test set, the first 130 observations not counted in experiment
    h_max: int
        Max prediction horizon
    n_iterations: int
        Number of iterations or size of test set
    motif_clustering_params: dict
        'beta': percentage of used patterns
        'mc_method': method of clustering: 'db' for DBSCAN, 'wi' for Wishart
    prediction_params: dict
        Prediction paramenters
    '''
    
    def exp_task(h, test_i, pm):
        predictor1 = TimeSeriesPredictor(clustered_motifs, non_pred_model)
        Y_preceding = Y2[:test_i + n_test_passed - h + 1]
        unified_preds, possible_predictions_list = \
            predictor1.predict(Y_preceding, h, **prediction_params)
        up, pp = predictor1.self_healing(Y_preceding, h, return_n_iterations=False, \
            unified_predictions=unified_preds, \
            possible_predictions=possible_predictions_list, \
            healing_logs_filepath=None,
            fixed_points_idx=[],
            **healing_params)
        # print(h, hs == h)
        i_h = np.argwhere(np.array(hs) == h).reshape(1, -1)[0][0]
        pm[i_h, test_i] = up[-1]
        if h_max > 0:
            for j in range(min(h, test_i + 1)):
                pm[h - j, test_i - j] = unified_preds[-(j + 1)]


        
    # clustered_motifs = predictor.cluster_motifs(Y1, **motif_clustering_params)
    try:
        mc_method = motif_clustering_params.get('mc_method', 'db')
        beta_str = str(int(motif_clustering_params.get('beta', 0.2) * 100))
        motifs_filename = 'motifs_' + mc_method + '_b' + beta_str + '.dat'
        with open(ROOT_PATH + 'data/motifs/' + motifs_filename, 'rb') as f:
            clustered_motifs = pickle.load(f)
    except FileNotFoundError:
        print('File with saved motifs not found!')
    if h_max > 0:
        prediction_matrix = np.empty((h_max + 1, n_iterations), dtype=object)

        args = list(zip([h_max] * n_iterations, list(range(n_iterations)))) \
            + list(zip(list(range(1, h_max)), [n_iterations - 1] * h_max))
    else:
        prediction_matrix = np.empty((len(hs), n_iterations), dtype=object)
        args = list(product(hs, list(range(n_iterations))))

    Parallel(n_jobs=1, backend="threading")(delayed(exp_task)(h0, i0, prediction_matrix) for (h0, i0) in args)
    
    if pm_filepath is not None:
        with open(pm_filepath, 'wb') as f: 
            pickle.dump(prediction_matrix, f)
        if logs_filepath is not None:
            with open(logs_filepath, 'a') as f:
                f.write(str(prediction_matrix))
    else:
        return prediction_matrix


def thrown_points_experiment(Y1, Y2, h_max, healing_params, logs_filepath, \
                             cluster_with_noise=True, step=1):
    '''Thrown points experiment

    Parameters
    ----------
    Y1: list or 1D np.ndarray
        Training set
    Y2: list or 1D np.ndarray
        Test set
    h_max: int
        Max prediction horizon
    healing_params: dict
        Healing parameters
    logs_filepath : str
        Path to logs file
    step : int
        Step of thrown points algorithm
    '''

    def thrown_points_exp_task(up, logs_file):
        predictor1 = TimeSeriesPredictor()
        predictor1.set_motifs(cm)
        predictor1.set_non_pred_model(non_pred_model)
        Y_preceding = Y1[-31:]
        to_delete = np.argwhere(up == 'N').reshape(1, -1)[0]   
        thrown = len(to_delete)

        if cluster_with_noise:
            possible_predictions_test = [np.array([up] * 20) + \
                np.random.normal(0, 0.01, 20) if up != 'N' \
                    else [] for up in up]
        else:
            possible_predictions_test = [np.array([up] * 20) if up != 'N' \
                    else [] for up in up]
                
        fixed_points_idx = np.delete(list(range(100)), to_delete).tolist()
        if not healing_params.get('fixed_points', True):
            fixed_points_idx = []

        unified_preds, possible_predictions_list, n_iterations = \
            predictor1.self_healing(Y_preceding, h_max, return_n_iterations=True, \
                unified_predictions=up, possible_predictions=possible_predictions_test, 
                healing_logs_filepath=None,
                fixed_points_idx=fixed_points_idx,\
                **healing_params)
        
        pickle.dump(thrown, logs_file)
        pickle.dump(unified_preds, logs_file)
        pickle.dump(n_iterations, logs_file)


    predictor = TimeSeriesPredictor()
    motifs_filename = 'motifs_' + healing_params.get('mc_method') + '_b' + \
        str(int(healing_params.get('beta') * 100)) + '.dat'
    with open(ROOT_PATH + 'data/motifs/' + motifs_filename, 'rb') as f:
        cm = pickle.load(f)
    predictor.set_motifs(cm)
    non_pred_model = predictor.create_non_pred_model(**healing_params)
    
    test_thrown_points = list(range(1, h_max, step))
    Y_thrown = []
    with open(ROOT_PATH + 'data/thrown_points_100.dat', 'rb') as f:
        thrown_points_ds = pickle.load(f)
    for thrown in test_thrown_points:
        Y_thrown.append(thrown_points_ds[thrown - 1])
        
    with open(logs_filepath, 'wb'):
        pass
    f = open(logs_filepath, 'ab')

    Parallel(n_jobs=-1, backend='threading')(delayed(thrown_points_exp_task)(Y_thrown[i], f) for i in range(len(Y_thrown)))
    f.close()