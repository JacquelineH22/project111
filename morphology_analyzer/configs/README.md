### Configuration: `test_cfg`

In the configuration files under `morphology_analyzer/configs`, the most important parameters in `test_cfg` are `score_thr` and `update_thr`:

- **`score_thr`**: controls the number of prompts predicted by the prompter. A higher `score_thr` will keep only more confident prompts, leading to fewer prompts passed to the SAM model.
- **`update_thr`**: controls the number of nuclei retained after SAM-based segmentation. A higher `update_thr` will filter out more low-confidence nuclei instances.

Based on our sensitivity analysis, setting `score_thr = 0.3` and `update_thr = 0.1` generally achieves the best performance across most datasets.