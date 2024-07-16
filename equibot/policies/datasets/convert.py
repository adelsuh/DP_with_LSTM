import numpy as np
import os

data_dir = "/home/mainuser/dp3_ws/dp3_setup_scripts/data_recording_(azure kinect)/data"
save_dir = "data"
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
for folder in os.listdir(data_dir):
  pc_file = np.load(os.path.join(os.path.join(data_dir, folder), "point_cloud.npy"))
  rgb_file = np.load(os.path.join(os.path.join(data_dir, folder), "rgb.npy"))
  action_file = np.load(os.path.join(os.path.join(data_dir, folder), "action.npy"))
  state_file = np.load(os.path.join(os.path.join(data_dir, folder), "robot_state.npy"))

  for record_t in range(pc_file.shape[0]):
    img_name = (f"converted_ep{int(folder.split('_')[-1]) :06d}_view{0}_t{record_t:04d}")
    save_path = os.path.join(save_dir, f"{img_name}.npz")
    np.savez(
      save_path,
      pc=pc_file[record_t],
      rgb=rgb_file[record_t],
      action=action_file[record_t],
      eef_pos=state_file[record_t],
    )