from dataclasses import dataclass

@dataclass
class DatasetConfig:
    name: str
    subjects: list
    sessions: list
    mapping_channels: dict
    mapping_non_eeg: dict
    event_id_map: dict
    event_classes: dict
    extra_info: dict = None  # Optional for additional dataset-specific info
