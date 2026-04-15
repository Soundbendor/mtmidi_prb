import os
import polars as pl
from util import util_main as UMN
from util import util_constants as UC
from util import util_rdb as UR
from util import util_optuna as UO
import sys

datasets = ['polyrhythms', 'dynamics', 'seventh_chords', 'mode_mixture', 'secondary_dominants']

emb_types = ["baseline-concat","baseline-chroma","baseline-mfcc","baseline-mel","musicgen-audio","musicgen-small","musicgen-medium","musicgen-large","jukebox",'MERT-v1-95M','MERT-v1-330M','wav2vec2-base','wav2vec2-large']

class_keys = ['loss', 'layer_idx', 'accuracy_score', 'f1_macro', 'f1_micro', 'balanced_accuracy_score', 'aic', 'aic_avg', 'bic', 'bic_avg', 'ebic', 'ebic_avg']

emb_idx = {k:i for (i,k) in enumerate(emb_types)}

num_emb = len(emb_types)

class Args:
    def __init__(self, expr_type, dataset, model_size, suffix):
        self.expr_type = expr_type
        self.dataset = dataset
        self.model_size = model_size
        self.suffix = suffix

def df_to_dict(cur_df):
    return {k:v[0] for (k,v) in cur_df.to_dict(as_series=False).items()}

def get_best_results(expr_type, dataset, model_size, suffix):
    args = Args(expr_type, dataset, model_size, suffix)
    cur_study = UO.create_or_load_study(args, seed=UC.SEED, evaluation = True)
    best_param_dict, best_trial_dict, attr_dict = UR.get_best_params_of_study(cur_study)
    cur_params = UR.make_eval_param_dict(best_param_dict, best_trial_dict)
    layer_idx = best_param_dict['layer_idx']['value']
    res_dir = UMN.by_projpath_multi(subpaths=['res', dataset, expr_type],make_dir = False)
    cur_csvf = os.path.join(res_dir, f'{model_size}_l{layer_idx}-{suffix}.csv')
    #print(cur_csvf)
    #cur_df = None
    cur_df = pl.read_csv(cur_csvf)
    return df_to_dict(cur_df)

def collate_best_results():
    suffix = 0
    metrics = {x:{'model': emb_types} for x in class_keys}
    for x in class_keys:
        for ds in datasets:
            if x != 'layer_idx':
                metrics[x][ds] = [-1.0 for _ in range(num_emb)]
            else:
                metrics[x][ds] = [-1 for _ in range(num_emb)]
    for ds in datasets:
        for m in emb_types:
            m_idx = emb_idx[m]
            res = get_best_results('linear', ds, m, suffix)
            for metric,val in res.items(): 
                metrics[metric][ds][m_idx] = val
    
    for metric,mdict in metrics.items():
        cur_df = pl.DataFrame(mdict)
        fname = f'{metric}_overall-{suffix}.csv'
        res_dir = UMN.by_projpath(subpath='res_overall', make_dir = True)
        cur_csvf = os.path.join(res_dir, fname)
        cur_df.write_csv(cur_csvf)

collate_best_results()



