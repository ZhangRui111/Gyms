MAX_EPISODES = 5
REPLY_START_SIZE = 100
UPDATE_FREQUENCY = 1

# Class Memory
M_EPSILON = 0.01  # small amount to avoid zero priority
M_ALPHA = 0.6  # [0~1] convert the importance of TD error to priority
M_BETA = 0.4  # importance-sampling, from initial value increasing to 1
M_BETA_INCRE = 0.001
M_ABS_ERROR_UPPER = 1.  # clipped abs error
