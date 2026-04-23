from stable_baselines3 import PPO
from ai_models.rl_environment import IrrigationEnv

env = IrrigationEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

model.learn(total_timesteps=100000)

model.save("ai_models/irrigation_agent")