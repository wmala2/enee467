from gymnasium.envs.registration import register

register(
    id="Rover-v0",
    entry_point="envs.tasks.rover_env:RoverEnv",
)

register(
    id="LineFollower-v0",
    entry_point="envs.tasks.line_follower_env:LineFollowerEnv",
)

register(
    id="LineFollowerReal-v0",
    entry_point="envs.tasks.line_follower_real_env:LineFollowerRealEnv",
)

register(
    id="GoalNav-v0",
    entry_point="envs.tasks.goal_nav_env:GoalNavEnv",
)
