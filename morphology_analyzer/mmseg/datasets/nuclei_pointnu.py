from .builder import DATASETS
from .nuclei_hv2 import NucleiHV2Dataset
import numpy as np


@DATASETS.register_module()
class NucleiPointnuDataset(NucleiHV2Dataset):
    def __init__(self, **kwargs):
        
        kwargs.setdefault("classes", None)
        kwargs.setdefault("type_color", )
        super(NucleiPointnuDataset, self).__init__(**kwargs)

    def result_to_inst(self, result):
        return result["inst_preds"].astype("int32")
    
    def get_inst_type(self, result, pred):
        inst_type = []
    
        for inst_id, label in result['inst_labels'].items():
            if label == -1:  # 如果类别是背景,则选择 inst_softmax 中的第二大类别
                softmax_probs = np.array(result['inst_softmax'][inst_id]) 
                top2_classes = np.argmax(softmax_probs[1:])
                inst_type.append(top2_classes + 1)
            else:
                inst_type.append(label + 1)   # gt里从0开始，我们也需要从0开始
        
        return np.array(inst_type, dtype=np.int32) 

    def prepare_test_img(self, idx):
        # 和train一样，获取gt
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    
