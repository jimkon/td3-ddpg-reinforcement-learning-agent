from mountain_car_stds import *
from td3 import *

agent = TD3(state_dims, action_dims, action_low, action_high)
run(agent, episodes=10, verbose=2, env=unwrapped_env)