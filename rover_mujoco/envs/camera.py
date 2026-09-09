import mujoco

# Geom groups the rover's own camera has to be told to draw
# The line tracks are group 3 and the goal-nav obstacles are group 4,
ONBOARD_GEOM_GROUPS = (3, 4)


def onboard_scene_option():
    """Scene options for any render that stands in for the rover's own camera
    Always pass this to Renderer.update_scene(). Without it the track and the
    obstacles are invisible to the policy."""

    option = mujoco.MjvOption()
    for group in ONBOARD_GEOM_GROUPS:
        option.geomgroup[group] = 1
    return option
