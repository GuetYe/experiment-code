# Common Setting for Networ awareness module.
from threading import Condition
# from ryu import cfg

import time

# CONF = cfg.CONF
COND_1 = Condition()   # detect save k_path
COND_2 = Condition()

# ---------------------------------- #
#              Ryu params            #
# ---------------------------------- #
SAVE_MAX_DATA_NUM = 5                # Save data
DISCOVERY_CANDIDATE_PERIOD = 5       # For discovering candidate path.
DETECT_TRAFFIC_MATRIX_PERIOD = 6     # For detect traffic matrix
DISCOVERY_PERIOD = 5                 # For discovering topology.
DISCOVERY_PATH_INTERVAL = 10         #
MONITOR_PERIOD = 6                   # For monitoring traffic
DELAY_DETECTING_PERIOD = 6           # For detecting link delay.
TOSHOW = False                       # For showing information in terminal
TO_MONITOR_SHOW = False              # For showing monitor infos in terminal
NODES = 14                           # number of switches
MAX_CAPACITY = 100000                # Max capacity of link kb/s
OBTAIN = 10                          # For obtain path infos
OBSERVE_TIME = 20
# WEIGT_FACTOR = [0.1,0.9]             # Weight factor
# IGNORE_PATH_LENGTH = 6

DROP_TOS = 192


# nodes = 14
# LINK_INFOS = {
#     (5, 9): 55000, (4, 7): 55000, (1, 3): 68000, (9, 4): 80000, (5, 6): 64000,
#     (9, 8): 25000, (8, 9): 25000, (6, 2): 25000, (11, 14): 100000, (5, 1): 56000,
#     (2, 5): 92000, (1, 11): 70000, (8, 5): 39000, (5, 8): 39000, (9, 1): 19000,
#     (8, 14): 50000, (13, 12): 29000, (3, 1): 68000, (4, 9): 80000, (6, 3): 42000,
#     (12, 13): 29000, (4, 13): 33000, (1, 5): 56000, (11, 1): 70000, (6, 5): 64000,
#     (3, 6): 42000, (4, 1): 60000, (13, 4): 33000, (10, 9): 45000, (9, 7): 15000,
#     (6, 4): 76000, (5, 4): 88000, (2, 6): 25000, (11, 4): 77000, (12, 14): 67000,
#     (4, 5): 88000, (10, 13): 95000, (1, 4): 60000, (9, 10): 45000, (7, 5): 35000,
#     (1, 9): 19000, (8, 7): 37000, (13, 9): 100000, (4, 2): 55000, (13, 10): 95000,
#     (4, 11): 77000, (7, 9): 15000, (14, 11): 100000, (9, 13): 100000, (4, 6): 76000,
#     (5, 2): 92000, (5, 7): 35000, (14, 8): 50000, (7, 4): 55000, (14, 12): 67000,
#     (9, 5): 55000, (7, 8): 37000, (2, 4): 55000
# }






