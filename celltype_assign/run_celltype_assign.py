from celltype_assign import CelltypeAssign
from celltype_assign.set_config import CelltypeAssignConfig

import yaml


# setup an assigner
config_yaml_file = './configs.yaml'

with open('./configs.yaml', 'r') as f:
    configs_data = yaml.safe_load(f)
configs = CelltypeAssignConfig(**configs_data)
assigner = CelltypeAssign(configs)

# run
assigner.preprocess()
assigner.celltype_assign()

print('Done!')
