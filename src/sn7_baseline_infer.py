import solaris as sol
import os
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
cm_dir = os.path.join(inference_top_dir, 'change_maps')
#change_map_from_masks(output_dir, cm_dir)
#group_pred(inference_top_dir)
score(cm_dir, '/app/spacenet7/train/L15-0331E-1257N_1327_3160_13/change_maps')