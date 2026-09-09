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
    - objects/  # All additional assets that will interact w/the scene are here
    - robots/   # All robot assets are here
  - docs/ # Explanation of setting up environments, additional info, are here
  - envs/
    - tasks/
      - __init__.py  	   # A file to make this directory visivble
      - line_follower_env.py  # A file to create a RL environment
    - __init__.py # Registration of all envs for Gymnasium is here
    - camera.py   # A file to handle the camera sensor
    - motor.py    # A file to handle higher fidelity motor information
    - tracks.py   # A file to handle a variety of line following tracks
  - scripts/  	# Directory to handle python scripts that run our environments
  - wandb/    	# Weights & Biases info/runs is stored here
  - README.md 	# Upper level information file to detail the project
  - pyproject.toml  # Additional packages are stored here
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

Everything above turns into `envs/tasks/line_follower_env.py`, a Gymnasium environment registered in `envs/__init__.py` under the id `LineFollower-v0`. Registration is what lets a training script call `gym.make("LineFollower-v0")` instead of importing the class, and it's also where a second variant of the same task gets its own id.

The constructor picks a track, compiles the scene once, and builds an offscreen renderer for the onboard camera:

```python
self.model = mujoco.MjModel.from_xml_path(model_path)
self.data = mujoco.MjData(self.model)
self.renderer = mujoco.Renderer(self.model, height=CAM_RES, width=CAM_RES)
```

MuJoCo compiles a model once and the tracks are separate scene files, so the track is fixed for the lifetime of the env instance rather than resampled on every reset. Pass `track=` to pin it. That argument matters more than it looks: without it an evaluation silently measures whichever track the constructor happened to draw, and the tracks aren't equally hard. The same policy scored 100% on one and 86.7% on another purely from that draw.

**Observation space.** The policy sees a 64×64 grayscale frame from the rover's own camera and nothing else:

```python
self.observation_space = spaces.Box(low=0, high=255, shape=(CAM_RES, CAM_RES, 1), dtype=np.uint8)
```

This is the part that decides whether the policy can leave simulation. The env knows the rover's exact position, because `self.data.qpos` is right there, and feeding it to the policy would make training much easier and the result worthless: nothing on the physical rover produces that number. The resolution is small on purpose, since rendering a camera every control step is what makes these rollouts far slower than a vector-observation task.

**Action space.** Two wheel velocity commands, written straight into MuJoCo's actuator controls:

```python
self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
```

One caution from training a later variant of this env. If you express the action in physical units, the policy has to explore in those units too, and SB3's Gaussian policy starts with a standard deviation of about 1 around zero. With a real motor dead zone in the mix, almost every sampled command lands somewhere that produces no motion at all, and the episode length pins itself to whatever your stuck detector allows. Normalizing the action to [-1, 1] and scaling inside `step()` avoids that.

**Control rate.** The environment steps physics at the scene's 2 ms timestep but only makes a decision at `CONTROL_HZ`, decimating the rest:

```python
CONTROL_HZ = 10.0
self._decimation = max(1, round(1.0 / (CONTROL_HZ * self._physics_dt)))
```

10 Hz is the real command rate from the Physical Constraints above, so this is one of those places where a hardware limit has to be modeled rather than approximated. It is also worth doing for a reason that only shows up in training. An earlier version let PPO pick a new wheel velocity every 2 ms, and it found a policy that whipsawed the command fast enough to vibrate in place. Net displacement stayed near zero, so image error stayed near zero, so reward was already about 97% of maximum in the first rollout and there was never any pressure to learn to drive.

**Reward.** Three terms, weighted so that finishing the track dominates:

```python
PROGRESS_WEIGHT = 100.0
CENTER_WEIGHT = 0.3
COMPLETION_BONUS = 50.0
```

Progress is forward arc length along the track's waypoints, divided by the track's own length so a full lap is worth about `PROGRESS_WEIGHT` whether the track is 2 m or 3 m long. Centering is the smaller shaping term that keeps the rover from cutting corners while chasing progress, and losing the line costs a flat `-1.0` per step.

Progress uses ground truth position, which contradicts the observation rule above until you notice the asymmetry: reward is a training-time construct that doesn't exist at deployment, while the observation has to work on the real rover. Privileged reward is fine. Privileged observation is not.

**Termination and truncation.** Gymnasium splits these, and the split is the End Conditions discussion in code:

```python
terminated = tipped_over or finished
truncated = self._lost_steps >= MAX_LINE_LOST_STEPS or self._episode_steps >= MAX_EPISODE_STEPS
```

`terminated` means the episode reached a real outcome, good or bad, and the value function should treat it as the end of the world. `truncated` means you cut it off, so the value function should still bootstrap from the final state. Getting these backwards quietly teaches the policy that running out of time is a catastrophe worth avoiding.

`MAX_EPISODE_STEPS = 300` is 300 control decisions, which at 10 Hz is 30 simulated seconds. Count in control steps, not physics steps. An earlier version capped at 500 physics steps, which is one simulated second, nowhere near enough to drive anything, and standing still genuinely was the best available policy.

**Reset and domain randomization.** Each reset re-randomizes floor shade, line shade, light intensity, and a few millimeters of camera mount slop, then spawns the rover at the track's first waypoint facing the second. The line geoms are selected by name:

```python
self._line_geom_ids = [
    i for i in range(self.model.ngeom)
    if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("line_")
]
```

That used to select on geom group instead, which coupled a rendering concern to a semantic one and meant the track's group couldn't be changed without silently disabling the color jitter.

**Extracting the error.** `_line_error` thresholds against the frame's own mean brightness rather than a fixed value, so it survives the randomized shades above, and it weights the centroid by how many dark pixels each column holds over the bottom `GROUND_BAND` of the frame. Both details earn their place: averaging column indices over the whole frame counts a column the same whether it holds one stray pixel or fifty, and it mixes the line far ahead in with the line underfoot, so a curve drags the centroid back toward center.

**One MuJoCo gotcha.** `close()` releases the renderer's GL context, and skipping it leaves that context broken for the next `Renderer` built in the same process. A single training worker never notices, because it only ever makes one. Any script that constructs several environments in one process, such as an evaluation loop over multiple tracks, will render solid black from the second environment onward and look exactly like a domain randomization bug.

## Training

Before beginning, assert that W\&B is enabled through SB3 so that we can quantitatively and qualitatively study features by extracting images of those graphs and compariing/contrasting to other research.

[image1]: images/Resources/plugged_in_esp32.jpg

[image2]: images/Resources/platformio_run_task.png

[image3]: images/Resources/cad_rover_mujoco.png

[image4]: images/Resources/reinforcement_learning_loop.png