import json, gym
import numpy as np
from skimage.color import rgb2gray
from cv2 import resize

def process(frame, env):
    with open('game_configs.json') as config_file:    
        config = json.load(config_file)
    config = config[env]

    if config["process_frame"]:
        # Crop
        frame = frame[config["crop_x_min"]:config["crop_x_max"],:]
        frame = frame[:,config["crop_y_min"]:config["crop_y_max"]]
        # Convert to gray scale
        if config["rgb2gray"]:
            frame = rgb2gray(frame)
        # Resize
        frame = resize(frame, (config["resize_x"],config["resize_y"]))
        #expand dimensions to shape (resize_x,resize_y,1)
        frame = np.expand_dims(frame, axis=2)
    return frame

def getSpecs(environment):
    env = gym.make(environment)
    frame = env.reset()
    state_shape = np.shape(process(frame,environment))
    n_actions = env.action_space.n
    return state_shape, n_actions


