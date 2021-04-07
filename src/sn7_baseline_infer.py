import solaris as sol
import os
import numpy as np
from sn7_baseline_postproc_funcs import change_map_from_masks, group_pred, score
config_path = '../yml/sn7_baseline_infer.yml'
config = sol.utils.config.parse(config_path)
print('Config:')
print(config)

output_dir = config['inference']['output_dir']
# make infernce output dir

os.makedirs(output_dir, exist_ok=True)

inferer = sol.nets.infer.Inferer(config)
#inferer()

inference_top_dir = os.path.dirname(output_dir)
#group_pred(inference_top_dir)
grouped_dir = os.path.join(inference_top_dir, 'grouped')
score_lists = []
aois = os.listdir(grouped_dir)
for aoi in aois:
    print(aoi)
    aoi_path = os.path.join(grouped_dir, aoi)
    cm_path = os.path.join(aoi_path, 'change_maps')
    mask_path = os.path.join(aoi_path, 'masks')
    change_map_from_masks(mask_path, cm_path, months=6)
    label_cm_path = os.path.join('/app/spacenet7/train/', aoi, 'change_maps')
    print(label_cm_path)
    scores = score(cm_path, label_cm_path)
    score_lists.append(scores)
    print("Mean f1 score AOI", aoi, np.mean(scores))

flattened_scores = np.vstack(score_lists)
print(flattened_scores.shape)
print("Mean f1, presicion and recall score all AOIs", np.mean(flattened_scores, axis=0))