# Diffusion Policy with LSTM for learned task state tracking

Based on [this repo](https://github.com/yjy0625/equibot)

For each experiment, the configs are:
| base          | lstm_none.yaml     |
|---------------|--------------------|
| lstm          | lstm.yaml          |
| obj_encoder   | obj.yaml           |
| obj_condition | obj_condition.yaml |

To run training:
```
python -m equibot.policies.train --config-name lstm_none prefix=base data.dataset.path=equibot/policies/datasets/data
```
prefix decides the folder where the pth files will be saved.

To evaluate on a real robot:
```
python -m equibot.policies.realworld_runner --config-name lstm_none training.ckpt=base/ckpt01999.pth
```

Changed files are mainly in equibot/policies: agents/dp_agent.py, agents/dp_policy.py, config files in equibot/policies/configs, train.py, realworld_runner.py (for running on the UR arm), datasets/dataset_new.py (for the lstm version), utils/diffusion/simple_conditional_unet1d.py (scaled down UNet to save computation), vision/pointnet_encoder_new.py (for obj_encoder).

Other new files (calc_plane.py, get_obj_pc.py, etc) are mainly supporting code to get things set up.