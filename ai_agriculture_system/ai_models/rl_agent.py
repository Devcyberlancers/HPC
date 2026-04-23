import torch
import numpy as np
import os
from stable_baselines3 import PPO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "irrigation_agent")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PPO.load(MODEL_PATH, device="cpu")


def get_actions(states):

    states = np.array(states, dtype=np.float32)

    # ⚡ batch predict
    actions, _ = model.predict(states)

    return [float(a) for a in actions]