from gymnasium.envs.registration import register

register(
    id="Rover-v0",
    entry_point="envs.tasks.rover_env:RoverEnv",
)


register(
    id="GoalNav-v0",
    entry_point="envs.tasks.goal_nav_env:GoalNavEnv",
)


register(
    id="LineFollowerPPO-v0",
    entry_point="envs.tasks.line_follower_ppo_env:LineFollowerPPOEnv",
)
