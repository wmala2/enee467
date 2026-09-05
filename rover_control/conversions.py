def convert_angular_vel_to_linear_vel(wheel_vel, wheel_radius):
    # Given wheel velocity in radians/sec, we need to get the meters/sec
    meters = wheel_vel * wheel_radius
    return meters


def convert_linear_vel_to_angular_vel(wheel_vel, wheel_radius):
    # Given wheel velocity in meters/sec, we need to get the radians/sec
    omega = wheel_vel / wheel_radius
    return omega
