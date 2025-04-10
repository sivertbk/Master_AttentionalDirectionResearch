from utils.dataset_config import DatasetConfig
from utils.helpers import format_numbers


# =============================================================================
#               Touryan et al. (2022) EEG Dataset Configuration
# =============================================================================

subjects = format_numbers(list(range(1, 21)),2)  # convert to list of strings with leading zeros like ["01", "02", ...]
sessions = [1] # Only one session
task_orientation = "external"



# Creating event id for each task and practice
event_id = {"1111": "Start of left perturbation",
            "1112": "End of left perturbation",
            "1121": "Start of right perturbation",
            "1122": "End of right perturbation",
            "1211": "Start driving car forward.",
            "1212": "End driving car forward.",
            "1331": "Beginning of sound clip relevant to task.",
            "1332": "End of a sound clip relevant to task.",
            "1341": "Beginning of a sound clip unrelated to task.",
            "1342": "End of a sound clip unrelated to task.",
            "1351": "Beginning of sound clip on mindfulness breath counting exercise.",
            "1352": "End of sound clip on mindfulness breath counting exercise.",
            "2221": "Start oncoming traffic with up to 12 vehicles driving south.",
            "2222": "End oncoming traffic with up to 12 vehicles driving south.",
            "2241": "Start vehicle being overtaken from front with up to 3 vehicles that appear in front driving north in lefthand of northbound lanes.",
            "2242": "End vehicle being overtaken from front with up to 3 vehicles that appear in front driving north in lefthand of northbound lanes.",
            "2251": "Start vehicle being overtaken from behind with up to 2 vehicles that appear behind the subject and drive past.",
            "2252": "End vehicle being overtaken from behind with up to 2 vehicles that appear behind the subject and drive past.",
            "2621": "Start showing a 45 MPH sign.",
            "2622": "End showing a 45 MPH sign.",
            "2631": "Start showing a 35 MPH sign.",
            "2632": "End showing a 35 MPH sign.",
            "2811": "Start police car in driver line of sight.",
            "2812": "End police car in driver line of sight.",
            "3111": "Start of a trial",
            "3112": "End of a trial",
            "3200": "All at high-perturbation rate with low visual complexity.",
            "3310": "Scenario is a straight road",
            "4210": "Vehicle moves into the defined lane of travel from either the left or right side of the lane",
            "4220": "Vehicle moves right of the defined lane of travel",
            "4230": "Vehicle moves left of the defined lane of travel",
            "4311": "Driver starts correcting lane position.",
            "4312": "Driver has completed the lane position correction.",
            "4411": "Vehicle approached within 3 meters of obstacle resulting in near miss.",
            "4421": "Vehicle collided with an environmental entity.",
            "4611": "Start of an undefined button press.",
            "4612": "End of undefined button press.",
            "4621": "Start of button press to acknowledge a police car.",
            "4622": "End of button press to acknowledge a police car.",
            "4710": "A single police car was in view, and reported, then occluded, then in view again, and reported again. It\u2019s not clear whether the subject knew it was the same target both times.",
            "4720": "Participant gives false alarm target response.",
            "4730": "Participant gives valid detection of target. ",
            "4740": "Participant gives repeated target detection response."
        }

DATASET_CONFIG = DatasetConfig(
    name="Touryan et al. (2022)",
    f_name="touryan2022",
    extension="set",
    task_orientation=task_orientation,
    subjects=subjects,
    sessions=sessions,  
    mapping_channels={},  
    mapping_non_eeg={},   
    event_id_map=event_id,
    state_classes={
        "focused": [], # events related to focused attention
        "mind-wandering": [], # events related to mind-wandering
    },
    task_classes={
        "police car": [], # events related to police car
        "collision": [] # events related to collision
    }
)