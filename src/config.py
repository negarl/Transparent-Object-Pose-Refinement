import os

tracebot = False
if tracebot:
    PATHS_DATASET_TRACEBOT = [
        "/home/negar/Documents/Tracebot/Tracebot_Negar_2022_08_04",
        "/home/thesis_code/Tracebot_Negar_2022_08_04",
        "/media/dominik/FastData/datasets/Tracebot_Negar_2022_08_04",
    ]
    PATH_DATASET_TRACEBOT = [path for path in PATHS_DATASET_TRACEBOT if os.path.exists(path)][0]
    PATH_REPO = '/'.join(os.path.dirname(__file__).split('/')[:-1])
else :
    PATHS_DATASET_BOP = [
        "/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_test_primesense_bop19/test_primesense_edited",
    ]
    PATHS_DATASET_BOP = [path for path in PATHS_DATASET_BOP if os.path.exists(path)][0]
    PATH_REPO = '/'.join(os.path.dirname(__file__).split('/')[:-1])