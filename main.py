from mountain_car_stds import *
from td3 import *

agent = TD3(state_dims, action_dims)

run(agent, episodes=10, verbose=2, env=unwrapped_env)