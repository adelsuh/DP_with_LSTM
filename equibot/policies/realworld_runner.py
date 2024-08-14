from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from rtde_io import RTDEIOInterface as RTDEIO
import robotiq_gripper
from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from threading import Thread, Event
from collections import defaultdict
import numpy as np
import time
from equibot.policies.utils.misc import get_agent
import pyk4a
from pyk4a import Config, PyK4A
import tqdm

## Parameters that you might want to adjust 
## ROBOT_HOST -> The robot's IP Address
## SCALE_FACTOR -> To increase/decrease the robot velocity 
## The acceleration in rtde_c.speedL() -> If there is any latency in robot movement (increasing acceleration = increasing deceleration)

# Define robot parameters
ROBOT_HOST = "192.168.20.25"  # IP address of the robot controller
SCALE_FACTOR = 0.3 # Scale factor for velocity command

def start_camera(self): 
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            camera_fps=pyk4a.FPS.FPS_30,
            depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
            synchronized_images_only= True,
        )
    )
    k4a.start()
    # Set white balance
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500

    return k4a

@hydra.main(
        version_base=None,
        config_path="configs",
        config_name="fold_synthetic"
)
def main(cfg):
    #Start camera (point cloud)
    k4a = start_camera()

    #Load agent
    agent = get_agent(cfg.agent.agent_name)(cfg)
    agent.load_snapshot(cfg.load_ckpt)

    # Initialize RTDEControlInterface
    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_io = RTDEIO(ROBOT_HOST)
    
    print("Creating gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect(ROBOT_HOST, 63352)
    print("Activating gripper...")
    gripper.activate()
    gripper_position = gripper.get_current_position()
    gripper_max = gripper.get_max_position()
    gripper_min = gripper.get_min_position()

    eval_episodes = 10
    try:
        for episode_idx in tqdm.tqdm(range(eval_episodes), leave=False, mininterval=5.0):
            if rtde_r.getRobotMode() == 7:
                obs = {}
                state = np.array(rtde_r.getActualQ())
                gripper_state = np.array([gripper.get_current_position()])
                obs["eef_pos"] = np.concatenate((state, gripper_state))

                capture = k4a.get_capture()
                if capture is not None:
                    points = capture.depth_point_cloud.reshape((-1, 3))

                    # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
                    bbox = [-500,-500,-600,1000,250,1200]
                    min_bound = np.array(bbox[:3])
                    max_bound = np.array(bbox[3:])  

                    #crop point clouds
                    indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
                    points = points[indices]
                obs["pc"] = points

                ac = agent.act(obs)
                
                #send command to robot 
                rtde_c.moveJ(ac, acceleration = 1.5) #adjust the acceleration if required 

                #get TCP velocity of robot
                # actual_velocity = rtde_r.getActualTCPSpeed()
                # actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity] #filter out extremely small numbers
                # print("Current velocity vector: " , actual_velocity)

                #get TCP pose of robot
                #actual_pose = rtde_r.getActualTCPPose()
                #print(actual_pose)
  
                # gripper.move(gripper_position, 155, 255)

                if gripper.is_gripping(): 
                    print("Gripping object")
                
                else: 
                    print("Not gripping object")

                #wait awhile before proceeding 
                time.sleep(1/100)

            else:
                print("Robot is not ready.")
                time.sleep(1)  # Wait longer if robot is not ready

    except KeyboardInterrupt:
        # Handle graceful shutdown here
        print("Stopping robot")
        rtde_c.stopScript()
        sm.stop()

if __name__ == "__main__":
    main()
