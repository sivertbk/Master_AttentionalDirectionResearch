from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str
    f_name: str # Name of the file containing the data
    extension: str
    task_orientation: str
    subjects: list
    sessions: list = None
    tasks: list = None
    states: list = None
    runs: list = None
    mapping_channels: dict = None
    mapping_non_eeg: dict = None
    event_id_map: dict = None
    event_classes: dict = None
    state_classes: dict = None
    task_classes: dict = None
    path_root: str = None
    path_raw: str = None
    path_epochs: str = None
    path_psd: str = None
    path_derivatives: str = None
    extra_info: dict = None  # Optional for additional dataset-specific info
