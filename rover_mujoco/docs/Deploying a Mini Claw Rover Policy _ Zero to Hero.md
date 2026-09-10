# **Deploying a Mini Claw Rover Policy : Zero to Hero**

## **Overview**

We’d like to test the capabilities of reinforcement learning (RL) from a simulator like [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html) to the real world on a cheap platform. This is a tutorial going from nothing to deploying policy and having it work. This tutorial will work from scratch, but it’s highly preferable to have some background experience with the rover before proceeding.

## **Getting Started**

### **Prerequisites (Hardware)**

- 1x Assembled Mini Claw Rover  
- USB-A to USB Micro B Data Transfer Cable

### **Prerequisites (Software)**

- [VSCode](https://code.visualstudio.com/)  
- [Python3](https://www.python.org/downloads/), any version from 3.10-3.14  
- [Git](https://git-scm.com/install/)  
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### **Workspace Setup**

You’ll want to create a workspace to bring all of the necessary tools into, in a homogenous format. We’ll clone some common projects into this workspace and use a [python virtual environment](https://docs.python.org/3/library/venv.html) (venv) to manage the dependencies in this project. Since we’re using a Python workspace, we should be system agnostic and the whole setup should work on Windows, Linux, and Mac.

Create the workspace and clone some repositories:

```shell
# Go to your desired install location
cd ~/Documents
mkdir mini_claw_ws
cd mini_claw_ws/
git clone https://github.com/wmala2/rover-firmware.git -b main
git clone https://github.com/wmala2/rover_camera_firmware.git -b main
git clone https://github.com/CursedRock17/matrix_lab_rover_above.git -b main
```

We should now be able to handle the rest of the project in a VSCode window. Note, we’ll still be typing commands into terminal windows, they’ll likely just be integrated into VSCode.

### Project Structure

For this project, we mainly care about the `rover_mujoco` directory which will handle all of the RL setup. It has the following structure:

```
rover_mujoco/
  - assets/
    - objects/  # All additional assets that interact with the scene are here
    - robots/   # All robot assets are here
  - docs/       # Environment setup, design notes, and measured results
  - envs/
    - tasks/
      - __init__.py                 # Makes this directory a package
      - line_follower_ppo_env.py    # The RL environment
    - __init__.py       # Registration of all envs for Gymnasium is here
    - contract.py       # The observation/action layout shared with the real rover
    - camera.py         # Extracts the three line centroids from an image
    - pid.py            # The classical controller the RL policy is measured against
    - line_scene.py     # Builds the rover + track scene in memory
    - line_dynamics.py  # BAM motor model and domain randomization
    - motor.py          # Higher fidelity motor information
    - tracks.py         # A variety of line following tracks
  - scripts/          # Python scripts that run our environments
  - tests/            # Regression tests, all CPU, no GPU needed
  - wandb/            # Weights & Biases info/runs is stored here
  - README.md         # Upper level information file to detail the project
  - pyproject.toml    # Additional packages are stored here
```

## **Flashing Firmware**

*If* you’ve already been given a pre-flashed rover you can skip this part. 

If this is the first time the rover+camera is being set up, you’ll have to flash the firmware with your settings. From the top level `mini_claw_ws` we’ll want to open up VSCode using the `code .` command. Trust the author of the folder and install the “PlatformIO” installer. Head to the “Extensions” tab on the left side bar. Search “PlatformIO IDE” and install, the logo looks a bit like an alien. Restart VSCode.

### Flashing the Rover

We need to use the ESP32 flashing plugin in VScode to correctly flash our ESP32 on the rover. Before flashing, make sure you’ve got the correct wifi interface setup in [wifi\_config.h](https://github.com/wmala2/rover-firmware/tree/main/include/wifi_config.h) part of the Rover. The IP that will be assigned to the Rover is within [control\_server.cpp](https://github.com/wmala2/rover-firmware/tree/main/src/control_server.cpp), you should be able to leave this as is. Navigate to the [Platform.IO](http://Platform.IO) logo on the left side bar, click it, and select the `rover-firmware` directory as your desired project since it has a `platformio.ini` file in it. Plug the USB cable into your rover from your laptop/computer.

![Rover ESP32 plugged in over USB][image1]

We now need to build our firmware, this can be done two ways in VSCode. You could either create a new terminal then run `platformio run`, or you can do the more standard setup. Navigate to the command palette bar at the top of the screen. Click “Run Task” \> “PlatformIO” \> “PlatformIO: Build”  
![PlatformIO build task in the VSCode command palette][image2]  
This should build the firmware for you. Then from that same location run “PlatformIO: Upload”. Once you’ve uploaded firmware, with the rover still plugged in you should be able to ping the given ip address from a terminal in VSCode filing in your rover’s IP address: `ping 192.168.50.201` or similar.

### Flashing the Camera

Now that our rover is setup, we should flash the cloned [rover\_camera\_firmware](https://github.com/wmala2/rover_camera_firmware) by opening the rover\_camera\_firmware and repeating the same process as the rover, but for the camera  

## **Environment Setup**

Now it’s time to start our setup of our MuJoCo simulation environment. Navigate to the [matrix\_lab\_rover\_above](https://github.com/CursedRock17/matrix_lab_rover_above/tree/mujoco_rover) section of the workspace in VSCode. As per the [installation instructions](https://github.com/CursedRock17/matrix_lab_rover_above/blob/mujoco_rover/README.md#repository-setup) we should open a terminal in VSCode, sync the project for our laptop, and test that MuJoCo correctly works with our rover. So for my CPU only laptop:

```shell
uv sync --extra cpu
uv run rover_mujoco/scripts/teleop_rover.py
```

This should open a MuJoCo 3D viewer with the latest rover MJCF file in it, with the capability to manually drive the rover around using your keyboard.  
![The CAD rover loaded in the MuJoCo viewer][image3]  
Now it’s time to build our setup for the RL training in simulation. *Note:* This will likely already be in the repository, but I’m going through the steps of setting up the environment, so that you know what to do. You should be able to extend this knowledge to your own environments.

The RL problem we’ll be tackling is a line follower with our robot because the problem has a relatively simple classical solution, is quite realistic in its goals, and capable of being ported to real life.

### Crucial Libraries

As mentioned before, this repository is managed by `uv` which will allow you to sync the project to the `pyproject.toml` file, but there are some important libraries that we want out of the box. I’ll put the entire file, then go over the libraries used

```xml
[project]
name = "rover_mujoco"
version = "0.1.0"
description = "RL Rover setup for MuJoCo and Gymnasium"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "mujoco>=3.1.0",
    "gymnasium>=0.29.1",
    "stable-baselines3>=2.3.0",
    "tensorboard>=2.15.0",
    "numpy>=1.24.0",
    "matplotlib>=3.8.0",
    "wandb",
    "scipy>=1.11.0",
    "onshape-to-robot>=1.8.3",
    "better-actuator-models>=1.0.2",
    "huggingface-hub>=1.30.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["envs"]
```

#### Mujoco

The MuJoCo [Python API](https://mujoco.readthedocs.io/en/stable/python.html) that’s necessary to hook into the MuJoCo system by writing custom scripts.

#### Gymnasium

The Gymnasium [Python API](https://pypi.org/project/gymnasium/) which allows us to easily integrate RL training into our workflow

#### Stable-Baselines3

Stable-Baselines3 ([SB3](https://pypi.org/project/stable-baselines3/)) contains out of the box RL policies which is easier than creating our own.

#### Weights & Biases

Odds are you’re going to want to quantitatively and qualitatively analyze how well you training went/goes, so we should have the means to do that.

[Weights & Biases](https://pypi.org/project/wandb/) should already be part of the installed packages, otherwise, be sure to add it to your `uv` setup. It records the model, the metrics, and the hyperparameters you used for a run, and it has a direct SB3 integration, so wiring it in is a callback rather than a rewrite. You’ll need to create an account, then make an [API Key](https://wandb.ai/authorize?ref=models)

In your `uv` synced environment run `wandb login` to access your account in the repo

### Creating Assets

In simulation, assets are the varying objects that will be affected by physics and interact with each other. Our assets are stored in `xml` format. We’ll be handling the assets here, but there’s a [Menagerie](https://mujoco.readthedocs.io/en/stable/models.html) containing tons of pre-made robots and similar assets that MuJoCo provides. For sake of completeness we’ll be creating one asset from scratch and one other with the [OnShape-to-Robot](https://pypi.org/project/onshape-to-robot/) plugin. Of course, these assets will already be in the repository when you get it, so you won’t have to create them unless you want to.

#### Simple Track

We’re going to want multiple tracks for training our rover, so the one we’ll start with is a simple circle track. Under the [assets directory](https://github.com/CursedRock17/matrix_lab_rover_above/tree/mujoco_rover/rover_mujoco/assets/objects), you’ll want to create a tracks folder, where we can add a circle track. TODO: Explain using the track

#### Complex CAD-designed Track

We can also design our tracks or similar assets in a CAD software like [OnShape](https://www.onshape.com/en/) which is the formal way of going about designs. TODO: OnShape-to-CAD tutorial reference.

## Environment Creation

It’s important to note the [Project Structure](#project-structure) before beginning, but essentially we’re going to want to create a script in [scripts](https://github.com/CursedRock17/matrix_lab_rover_above/tree/mujoco_rover/rover_mujoco/scripts)/ to run our RL training and deployment. We’ll want to reproduce/adjust the rover setup and environment, so we’ll create modular pieces in [envs/](https://github.com/CursedRock17/matrix_lab_rover_above/tree/mujoco_rover/rover_mujoco/envs), then create and register our main environment. 

### Preparation

It’s important to also **frame the problem** before beginning the actual layout as you most aptly solve the problem in a reproducible manner. This is going to allow us to develop a more robust policy, make changes more easily in the future and have modular blocks. 

**Objective:** Continuously follow/stay on various black line tracks as stable as positive.  
**Subgoals:** Drive as fast as the rover can physically move and stay as central to that line as possible.

We can only complete the subgoals if we’re able to complete the main goal, but those are pieces of information we still want to have in mind if we’re to make a more robust policy. 

Now that we have our main objective, we need to consider the physical constraints of this system as ultimately we’d like to deploy this on a real rover and these contribute to what factors we can consider when improving the policy. These factors will contribute to **[action](https://gymnasium.farama.org/introduction/basic_usage/#action-and-observation-spaces)** [and **observation**](https://gymnasium.farama.org/introduction/basic_usage/#action-and-observation-spaces) spaces that are available to our Gymnasium agent which play into the lifecycle of the RL loop

![The reinforcement learning loop][image4]

**Observation Space:** These are the variables that our agent (the rover) is able to acquire from the overall **state** space, where the state space would be every possible variable in an environment. The physical rover has three sensors at its disposal: ESP32 Camera, Right Motor Encoder, Left Motor Encoder. Now, we can either take these sensors at face value, or we can extrapolate concepts from those sensors. From the camera, we’re interested in pulling various centroids of the line. From even a lower resolution image, we should be able to locate the edges of the track based on large deviations of colors from a somewhat lighter floor to the black of the taped line. We can then find the centroid close, mid-distance, and far away if we’re tilting the camera say 45 degrees. Those 3 XY points will be essential in learning patterns based on relative position to the line, especially since the camera is in the middle of the rover’s X-axis, thus the side wheels are evenly spaced from the camera. We can then expand upon motor encoder counts to acquire estimated velocity of the rover, by acquiring the current encoder counts and deriving them, by subtracting the previous encoder counts and dividing by the change in time between now (time t=0) and the last timestep (time t=t-1). This also means that odometry drift hopefully won’t accumulate as poorly which shouldn’t even matter as the camera will constantly reference the line. The theme though is that we shouldn’t estimate data that we don’t have, for instance we can get absolute position of the rover in simulation, but we don’t have access to that information in the real world and since the policy would rely and learn to rely on that data, it would fail as soon as it got to the real world.

**Action Space:** These are the variables that our agent (the rover) is able to manipulate to force a reaction from the environment. In this case, the only thing that our policy can affect with the rover are speed commands. We can send velocities to each of the two motors which will in turn rotate the wheels. The physical rover does this through [closed-loop velocity](https://github.com/wmala2/rover-firmware#closed-loop-velocity-m) commands by providing a linear velocity to each wheel then acquiring that underneath the hood with encoder counts and a Proportional, Integral, Derivative (PID) controller. Thus, our action space must send velocity commands to the motors in simulation to mimic this real life behavior. Similarly, our action space can provide both a linear velocity (x) and angular velocity (z), which can then be converted to motor velocities based on a separate function that converts overall rover velocities to wheel velocities using the linear velocity (Vx), angular velocity (Wz), wheel radius (R ), and wheel separation (d) based on the forward kinematic equations for differential drive systems:

${V}_{left}=(2{V}_{x}-d{W}_{z})/2R$  
${V}_{right}=(2{V}_{x}+d{W}_{z})/2R$

**Reward Function:** The reward function dictates the actual “learning” that our agent will do and numerically drives the policy to respond in certain ways. The reward function can assign numerical context and positive/negative consequences to the rover based on those actions and how the environment responds. The reward function directly correlates with the objective and subgoals of the environment. In this case, we could normalize the concepts/variables that we will observe, then multiple by a gain value and add them up to get our desired reward. So breaking up problem, the most important objective is “completing” the track which means perpetually moving forward (side wheels to the rear, caster wheel \+ camera in the front), thus this value should have the greatest gain, losing value if we start to go backwards. It also means that if we’re moving faster (greater linear velocity) well get a higher reward, whereas moving slower, but still forward provides a positive reward, but not as much. The next most important thing is likely alignment, which is how central that camera is to the line. The will relate to the error of the centroid in multiple places, so the closest centroid should have the most weight, then a little less for the middle centroid, then the least for the centroid that’s furthest down the track. If our values our normalized from let’s say \[-1, 1\], we could take 1 and subtract the absolute value of the error of those centroids. So if camera is directly in the middle of the line (centroid\_error \= 0), then 1-0 \= 1, or the maximum reward (which is then multiplied by the gain), the minimum reward would be 1-1 \= 0, which would mean we’re completely not aligned. The segmentation into the 3 centroids would provide us with future prediction, but prioritize short term gain as that’s the most important. The last thing would be a “smoothness” value which is related to the angular velocity of the rover, as we’d like to be turning less wherever possible, it should be a very small penalty as we’re still going to need to turn at corners and edges, but while driving on straight segments, we’d rather not veer away as it will require more correction in the future. If there’s no line at all (centroid\_error \= 0), then we should start to rapidly lose points as we’ve gone way, way off track. The weird part is that if you start to accumulate some of these errors overtime you may accidentally drag your policy down, even though it’s doing overall super well.

**End Conditions:** There need to be concepts in place to end a certain episode based on failure or passing. For instance, if our episode score is better than our reward goal, we should terminate, register the score as a passing grade, then reset. If our episode fails based on some failure “reward” (numerical value where the policy is obviously going to continously get worse), then we should terminate, register the score as a failing grade, then reset. Additionally, if we lose the line too many times we should terminate, register the score as a passing grade, then reset. We should also have the ability to truncate an episode which wouldn’t necessarily register as a pass/fail, but instead end the episode for being too long based off of the current step and maximum episode we’d want to see per episode. This is a bit weird, but if the policy isn’t passing, but is taking a really long time, we wouldn't necessarily want to mark the system as a failure as it could just be really slow, but we also don’t want to provide emphasis that’ll work, cause it may not. After a certain number of episodes (say 100), we should compare our average score of all of our episodes to see if by statistical average, we’re doing better than roughly our reward goal. If the agent is constantly passing the objective, then we shouldn’t waste time training for longer and mark the problem as “solved”

### Physical Constraints

We’ll have to have some tolerances within the simulation, one tolerance is actually arriving at the goal (it won’t be very large), but once we get pretty close to the objective we may want to end the policy. Odds are, we’ll want to force the space to be a maximum of a 5 meter by 5 meter square to start. If we start to go too large, then the policy not only be harder to train, but we’ll start to lose quality in wheel odometry. Our rover isn’t very expensive and doesn’t have great sensors on board, that’s the point of these rovers, they’re meant to be affordable and easy to work with (more sensors means more knowledge to obtain and have a harder setup), but that means the fidelity of our sensors will decline. This leads to the a many DR parameters that might be useful, which of course we should document in tabular format with a parameter name, description, and min/max in our documentation in the project once it’s formalized.  
  \- Motor noise: Should use something like BAM to induce noise in our motors just like real life  
  \- Motor acceleration: Just like before, motors don’t automatically get to the desired speed, that’s why there’s a PID in the [wmala2/rover-firmware repository](https://github.com/wmala2/rover-firmware/blob/main/src/pid.cpp).  
  \- Friction: There will be friction between the rubber wheels and whatever material the floor is made of, which can change from concrete to grass to carpet to more, we can’t account for everything, so make the policy universally robust.  
  \- Camera brightness: There will be a certain amount of light based on the resolution and FOV of the physical camera that may make seeing obstacles easier/harder.  
  \- Gaussian noise: There will likely be some noise in our data, Gaussian in nature, that would be nice to very sensitively play with  
  \- Command Rate: It’s important to know that the maximum our motors can change will be at the control rate, which is pretty standardly 10Hz (once every 100ms), thus if the rover wheels in sim change faster than that, we’ll negatively influence the policy  
  \- Min/Max Speed: There’s a minimum and maximum speed that our rover can actually travel at, too slow and the wheels don’t have enough torque to overcome the friction interaction with the ground surface, too much and we physically cannot hit that speed or will pop wheelies (play it conservative).  
  \- Battery Discharge: Shouldn’t be a problem b/c we have documented history that the rover can last 8hrs minimum with that 3S battery.

### Writing the Environment

Everything above turns into `envs/tasks/line_follower_ppo_env.py`, registered in
`envs/__init__.py` as `LineFollowerPPO-v0`. Registration is what lets a training script call
`gym.make("LineFollowerPPO-v0")` instead of importing the class.

There are no per-track scene files. `envs/line_scene.py` builds the rover, camera and tape in
memory from a list of waypoints, so adding a track means adding a function to `envs/tracks.py`
and naming it in `TRACKS`:

```python
TRACKS = {
    "circle": circle_waypoints,
    "figure8": figure8_waypoints,
    "goomba": goomba_waypoints,
    "oval": oval_waypoints,
}
```

Each environment instance is pinned to one track for its lifetime, because MuJoCo compiles a
model once. Training runs one worker per track in parallel rather than one track after
another, so every gradient update sees all of them.

**Observation space.** Eleven floats, all in `[-1, 1]`:

```python
self.observation_space = spaces.Box(-1, 1, shape=(11,), dtype=np.float32)
```

Indices 0-2, 3-5 and 6-8 are the near, middle and far camera bands, each as (centroid x,
centroid y, visible). Indices 9-10 are the two wheel speeds divided by 10 rad/s. That is the
whole input: no position, no yaw, no track identity, nothing the real rover cannot measure.

The important part is where it is built. Both the simulator and
`rover_control/rl_rover.py` call the same `observation_from_sensors` in `envs/contract.py`,
one from a rendered frame and one from the ESP32 camera. Writing that twice is how sim and
hardware quietly drift apart until the policy behaves differently on the robot for reasons
nobody can find.

**Action space.** Two normalized commands, decoded by the same shared module:

```python
self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
forward, steering = np.clip(action, -1, 1) * [3.0, 4.0] + [3.0, 0.0]
```

So `[0, 0]` is the PID's cruise speed, `[-1, 0]` is a stop, and steering is a differential
added to both wheels. Keep actions normalized rather than in rad/s: SB3's Gaussian policy
starts with a standard deviation near 1 around zero, so an action space in physical units
means almost every sampled command lands somewhere useless, and with a motor dead zone in the
mix the policy never discovers that moving is possible.

**Control rate.** Physics runs at the scene timestep, but a decision is made only at
`CONTROL_HZ`:

```python
self.substeps = round(1 / CONTROL_HZ / self.model.opt.timestep)
```

10 Hz is the rover's real command rate, so this is a hardware limit that has to be modelled
rather than approximated. It also prevents a specific failure: given a decision every 2 ms,
PPO once found a policy that whipsawed the command fast enough to vibrate in place, which
kept image error near zero and paid almost full reward from the first rollout.

**Reward.** Progress along the track, scaled by how well centred the rover is:

```python
progress_reward = float(np.clip(delta / (CRUISE_RAD_S * WHEEL_RADIUS * self.dt), -2, 1))
motion_reward = progress_reward * (0.25 + 0.75 * alignment) if visible else -1.0
reward = motion_reward - smoothness_penalty - 0.01
```

Progress is normalized so cruise speed earns 1.0 and saturates there, which stops the policy
buying reward with speed it cannot control. Multiplying rather than adding the alignment term
means neither progress nor centring can be traded away for the other. Losing the line costs a
flat -1.0, the small smoothness penalty charges for jerky steering, and the -0.01 per step
discourages dawdling. Completion adds +10, any failure -5.

Progress uses ground truth position, which looks like it contradicts the observation rule
above. It does not: reward exists only during training, while the observation has to work on
the real rover. Privileged reward is fine, privileged observation is not.

**Termination and truncation.** Gymnasium separates these, and the split is the End Conditions
discussion in code:

```python
terminated = reason in ("tipped", "off_track", "line_lost", "completed")
truncated = reason == "timeout"
```

An episode terminates on tipping, leaving a 6 cm corridor, losing the line for half a second,
or finishing the lap. Running out of steps is a truncation. The difference matters to the
value function: `terminated` means no future reward exists, `truncated` means the episode was
cut short and it should still bootstrap. Getting these backwards teaches the policy that time
running out is a catastrophe to avoid.

**Reset.** Each episode starts at a waypoint with up to 1 cm of lateral offset and 5 degrees of
heading error, then settles for one simulated second before the first observation. `random_start`
picks the waypoint at random, which is what stops a policy learning only the first corner.

**Swappable dynamics.** The `dynamics` argument selects `"nominal"` velocity servos, `"bam"`
for the measured motor model, or `"dr"` to add domain randomization on top. Keeping these as
one switch means the sim2real ladder is a flag rather than a fork of the environment.

## Training

W\&B is already wired in, and it is on by default. `scripts/train_ppo.py` opens a run in the
project `rover-line-follower` under whatever account `wandb login` authenticated, mirrors every
TensorBoard scalar into it with `sync_tensorboard=True`, and at the end uploads the trajectory
plot and the selected checkpoint as an artifact. An authentication failure stops startup rather
than silently training without logs, which is deliberate: a long run you cannot compare
afterwards is close to worthless.

```shell
# Train on every track, logging to W&B.
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run scripts/train_ppo.py

# No account, or a quick local check.
uv run scripts/train_ppo.py --no-wandb --timesteps 128 --output runs/smoke

# Group related runs so W&B shows them on one axis.
uv run scripts/train_ppo.py --wandb-group camera-vs-chassis
```

Run the short `--no-wandb` version first, every time. It reaches the argument parsing, the
vector environment, the logger and the checkpoint writer in about thirty seconds, which is
where configuration mistakes actually live, and it costs nothing compared to discovering them
an hour into a real run.

[image1]: images/Resources/plugged_in_esp32.jpg

[image2]: images/Resources/platformio_run_task.png

[image3]: images/Resources/cad_rover_mujoco.png

[image4]: images/Resources/reinforcement_learning_loop.png