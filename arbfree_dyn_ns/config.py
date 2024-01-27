import logging
import os
from pathlib import Path

import numpy as np
from joblib import Memory

memory = Memory(location=Path(os.path.expanduser("~/.cache/arbfree-dyn-ns")))

BASE_PATH = Path(r"/home/jens/Nextcloud/Documents/GBS/Thesis")
PKL_BASE_PATH = BASE_PATH / "out-pkl"
DEFAULT_BUBA_PATH = BASE_PATH / "data" / "buba"
MAIN_GRID = np.array([1, 2, 3, 5, 6, 8, 10, 12, 15, 20, 25], dtype=float)

# set up logging to file
logging.basicConfig(
    filename=BASE_PATH / 'arbfree-dyn-ns.log',
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
