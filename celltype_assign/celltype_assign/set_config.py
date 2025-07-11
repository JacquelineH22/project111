from dataclasses import dataclass
from typing import Optional
import json
import os


@dataclass
class CelltypeAssignConfig:
    """Configuration class for CelltypeAssign
    """
    # output files
    spatial_info_json_file: str
    cell_info_json_file: str

    # input files
    nuclei_seg_json_file: str
    spatial_positions_file: str
    scalefactors_json_file: str
    deconvolution_file: str
    type_mapping_json_file: str
    nuclei_seg_types_json_file: str 

    # process parameters
    with_interval: bool = True
    spot_shape: str = 'circle'
    cell_seg_scale: float = 1
    header: bool = False
    is_proportion: bool = False
    cell_composition_mode: str = 'full'  # 'full' or 'top'
    top_m_count: int = 3
    load_from_file: bool = False
    metric: str = 'kl'
    top_k: int = 5
    cbi_threshold: Optional[float] = None
    n_processes: int = 10


    @classmethod
    def from_json(cls, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(**config_dict)


    def to_json(self, config_file: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

        return


    def validate(self):
        """Validate configuration parameters"""
        # validate cell composition mode
        if self.cell_composition_mode not in ['full', 'top']:
            raise ValueError('cell_composition_mode must be "full" or "top_m"')
            
        # validate top_m_count
        if self.cell_composition_mode == 'top' and self.top_m_count <= 0:
            raise ValueError('top_m_count must be positive when using top_m mode')

        # validate metric
        if self.metric not in ['kl', 'js', 'ce', 'cosine']:
            raise ValueError('Unsupported metric: %s' % self.metric)

        # validate CBI threshold
        if self.cbi_threshold is not None and self.cbi_threshold < 0:
            raise ValueError('CBI threshold must be non-negative')

        # validate number of processes
        if self.n_processes <= 0:
            raise ValueError('Number of processes must be positive')

        # validate input files exist
        required_files = [
            self.nuclei_seg_json_file,
            self.nuclei_seg_types_json_file,
            self.spatial_positions_file,
            self.scalefactors_json_file,
            self.deconvolution_file,
            self.type_mapping_json_file
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError('Required file not found: %s' % file_path)

        return
