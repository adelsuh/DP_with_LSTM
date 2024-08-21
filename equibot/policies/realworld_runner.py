from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface 
from rtde_io import RTDEIOInterface as RTDEIO
import robotiq_gripper
from threading import Thread, Event
from collections import defaultdict
import numpy as np
import time
from equibot.policies.utils.misc import get_agent
import pyk4a
from pyk4a import Config, PyK4A
import tqdm
import hydra
import torch
import pytorch3d.ops as torch3d_ops
import open3d as o3d

## Parameters that you might want to adjust 
## ROBOT_HOST -> The robot's IP Address
## SCALE_FACTOR -> To increase/decrease the robot velocity 
## The acceleration in rtde_c.speedL() -> If there is any latency in robot movement (increasing acceleration = increasing deceleration)

# Define robot parameters
ROBOT_HOST = "192.168.20.25"  # IP address of the robot controller
SCALE_FACTOR = 0.3 # Scale factor for velocity command

def start_camera(): 
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

def downsample_with_fps(points: np.ndarray):
        # fast point cloud sampling using torch3d
        points = torch.from_numpy(points).unsqueeze(0).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=1024)
        points = points.squeeze(0).cpu().numpy()
        points = points[sampled_indices.squeeze(0).cpu().numpy()]
        return points

def distance_from_plane(points):
    #define the plane equation (determined from plane segementation algorithm)
    a = 0.10
    b = 0.64
    c = 0.76
    d = -617.23
    #calculate distance of each point from the plane 
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    return distances

@hydra.main(
        version_base=None,
        config_path="configs",
        config_name="fold_synthetic"
)
def main(cfg):
    #Start camera (point cloud)
    k4a = start_camera()

    #Load agent
    cfg.data.dataset.num_training_steps = 1
    agent = get_agent(cfg.agent.agent_name)(cfg)
    agent.load_snapshot(cfg.training.ckpt)

    # Initialize RTDEControlInterface
    rtde_c = RTDEControlInterface(ROBOT_HOST)
    rtde_r = RTDEReceiveInterface(ROBOT_HOST)
    rtde_io = RTDEIO(ROBOT_HOST)
    
    print("Creating gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect(ROBOT_HOST, 63352, 5.0)
    print("Activating gripper...")
    gripper.activate()
    gripper_max = gripper.get_max_position()
    gripper_min = gripper.get_min_position()

    obs_horizon = cfg.model.obs_horizon
    try:
        pc_history = []
        eef_history = []
        actions = []
        if cfg.model.use_lstm:
            h = torch.zeros(1, 1, cfg.model.lstm_dim).to(cfg.device)
            c = torch.zeros(1, 1, cfg.model.lstm_dim).to(cfg.device)
            print(h.dtype)
        while True:
            print("======")
            if rtde_r.getRobotMode() == 7:
                print("Observing")
                state = np.array(rtde_r.getActualQ())
                gripper_state = np.array([gripper.get_current_position()])
                eef_history.append(np.concatenate((state, gripper_state)).reshape((1, 1, -1)))

                capture = k4a.get_capture()
                if capture is not None:
                    points = capture.depth_point_cloud.reshape((-1, 3))

                    # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
                    bbox = [-500,-500,-600,1000,400,1200]
                    min_bound = np.array(bbox[:3])
                    max_bound = np.array(bbox[3:])  

                    #crop point clouds
                    indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
                    points = points[indices]

                distances = distance_from_plane(points)
                points = points[distances > 7]
                points = downsample_with_fps(points)
                pc_history.append(points.reshape((1, 1, -1, 3)))

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pc_history[-1][0,0])
                # o3d.visualization.draw_geometries([pcd])
                # o3d.io.write_point_cloud("./data.ply", pcd)
                # break
                
                if len(actions) == 0 and len(pc_history) >= obs_horizon:
                    print("Computing predictions")
                    obs = dict(pc = torch.from_numpy(np.concatenate(pc_history[-1*obs_horizon:], axis=1)),
                               eef_pos = torch.from_numpy(np.concatenate(eef_history[-1*obs_horizon:], axis=1)))
                    
                    if cfg.model.use_lstm:
                        if len(pc_history) > obs_horizon:
                            obs_past = dict(pc = torch.from_numpy(np.concatenate(pc_history[:-1*obs_horizon], axis=1)),
                                eef_pos = torch.from_numpy(np.concatenate(eef_history[:-1*obs_horizon], axis=1)))
                            flattened_pc = obs_past["pc"][0].to(cfg.device, dtype=torch.float32)
                            z = agent.actor.ema.averaged_model["encoder"](flattened_pc.permute(0,2,1))["global"]
                            z = z.reshape(1, len(pc_history)-obs_horizon, -1)
                            z = torch.cat([z, obs_past["eef_pos"].to(cfg.device)], dim=-1).to(torch.float32)
                            _, (h, c) = agent.actor.ema.averaged_model["lstm"](z.permute(1,0,2), (h,c))
                        pred_actions, h, c = agent.act(obs, hidden=(h,c))
                    else:
                        pred_actions = agent.act(obs)
                    actions = pred_actions[0, :cfg.model.ac_horizon].tolist()
                    pc_history = []
                    eef_history = []
                
                if len(actions) > 0:
                    print("Taking action")
                    ac = actions.pop(0)
                    #send command to robot
                    rtde_c.moveJ(ac[:6]) #adjust the acceleration if required
                    gripper_position = int(ac[6])
                    if gripper_position < gripper_min:
                        gripper_position = gripper_min
                    elif gripper_position > gripper_max:
                        gripper_position = gripper_max
                    gripper.move(gripper_position, 155, 255) 

                #get TCP velocity of robot
                # actual_velocity = rtde_r.getActualTCPSpeed()
                # actual_velocity = [0 if abs(x) < 0.01 else x for x in actual_velocity] #filter out extremely small numbers
                # print("Current velocity vector: " , actual_velocity)

                #get TCP pose of robot
                # actual_pose = rtde_r.getActualTCPPose()
                # print(actual_pose)
  
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

if __name__ == "__main__":
    main()
