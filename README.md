# Orbis 2: A Hierarchical World Model for Driving
**Official Implementation**
## [Paper (TODO)](https://arxiv.org/abs/XXXX.XXXXX) | [Project Page](https://lmb-freiburg.github.io/orbis2.github.io/) | [HuggingFace Demo](https://huggingface.co/spaces/sud0301/orbis2_test) | [Orbis 1](https://lmb-freiburg.github.io/orbis.github.io/)

>[Sudhanshu Mittal*](https://lmb.informatik.uni-freiburg.de/people/mittal/), [Arian Mousakhan*](https://lmb.informatik.uni-freiburg.de/people/mousakha/), [Silvio Galesso*](https://lmb.informatik.uni-freiburg.de/people/galessos/), [Karim Farid](https://lmb.informatik.uni-freiburg.de/people/faridk/), [Johannes Dienert](https://lmb.informatik.uni-freiburg.de/people/dienertj/), [Rajat Sahay](https://lmb.informatik.uni-freiburg.de/people/sahayr/), [Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox/index.html)
> <br>University of Freiburg<br>
> <sub>* Main contributors</sub>

<!-- <table>
  <tr>
    <td><img src="imgs/teaser1.gif" width="100%"></td>
    <td><img src="imgs/teaser2.gif" width="100%"></td>
  </tr>
  <tr>
    <td><img src="imgs/teaser3.gif" width="100%"></td>
    <td><img src="imgs/teaser4.gif" width="100%"></td>
  </tr>
</table> -->

![Teaser](imgs/Framework.png)

Orbis-2 is a hierarchical driving world model that generates long-horizon future video conditioned on past frames and an optional **steering signal**. A frozen low-frame-rate **L2** predictor provides abstract long-range context, while the **L1** detail predictor autoregressively generates high-frame-rate future frames. Steering can be given either as raw ego-motion values (speed and yaw rate) or as a 2D trajectory that the model should follow.

## Installation
```bash
git clone https://github.com/lmb-freiburg/orbis2.git
cd orbis2
conda env create -f environment.yml
conda activate orbis2_env
```

## Checkpoints
Link to the [Checkpoints](https://huggingface.co/sud0301/orbis2_test) on Huggingface.
The checkpoints repository contains the necessary model weights and config files.

<!--
Each experiment directory contains the model config and its checkpoint: (TODO)
```
logs_wm/orbis2_stage2_450M_288x512_10hz/
├── config.yaml
└── checkpoints/
    └── last.ckpt
```

Move the downloaded checkpoint into the relevant experiment directory, e.g.:
```bash
mv last.ckpt logs_wm/orbis2_stage2_450M_288x512_10hz/checkpoints/
```
-->

## Steerable Video Generation (Roll-out)
`evaluate/rollout_demo_v2.py` rolls out the world model from a single input video: it samples the L1 (high-rate) and L2 (low-rate, further back in time) context windows directly from the video, then autoregressively generates future frames.

To roll out with trajectory steering and an ego-centric trajectory overlay:
```bash
python evaluate/rollout_demo_v2.py \
    --exp_dir logs_wm/orbis2_stage2_450M_288x512_10hz \
    --config config.yaml \
    --video /path/to/input_video.mp4 \
    --l1_frame_rate 10 \
    --num_steps 5 \
    --num_gen_frames 7 \
    --trajectory_file example_trajectory.csv \
    --vis_mode trajectory_ego \
    --output_dir rollout
```

To roll out **without steering** (unconditional generation), simply omit the steering arguments:
```bash
python evaluate/rollout_demo_v2.py \
    --exp_dir logs_wm/orbis2_stage2_450M_288x512_10hz \
    --config config.yaml \
    --video /path/to/input_video.mp4 \
    --l1_frame_rate 10 \
    --num_steps 5 \
    --num_gen_frames 60 \
    --output_dir rollout_uncond
```

Each rollout step predicts `model.num_pred_frames` future frames, so the total number of generated frames is `num_gen_frames × num_pred_frames`. Results are written to `--output_dir`: individual frames under `fake_images/sequence_XXXX/frame_XXXX.jpg` and one animated `rollout_XXXX.gif` per generated sequence.

### Steering inputs
The model supports three steering modes (the first two are mutually exclusive):

- **`--trajectory_file`**: a `.csv` or `.npy` file with `[T, 2]` rows of raw `(x, y)` trajectory points in meters (forward, lateral, local ego frame). The path is resampled by arc length to the rollout length and converted to speed / yaw-rate conditioning via finite differences — so a hand-drawn or planned path of any temporal resolution can be used directly.
- **`--steering_file`**: a `.csv` or `.npy` file with `[T, 2]` rows of raw `(speed, yaw_rate)` values, already at the odometry rate expected by the model.
- **Neither**: the rollout runs unconditionally (no steering).

An example trajectory is provided in `example_trajectory.csv`. Ready-made steering profiles (e.g. sharp left/right turns at fixed speed) can be found in `steering_values/`.

### Useful options
| Argument | Description |
|---|---|
| `--l1_frame_rate` | Frame rate (Hz) to sample the L1 context and generate at; must evenly divide the video's native frame rate. |
| `--start_frame` | Native-video frame index where the L1 context window starts (defaults to the latest window that fits). Enough video history must precede it for the L2 context. |
| `--num_steps` | Sampler steps (NFE) for the L1 predictor. Distilled models (`config_distill.yaml`) need only a few steps (e.g. 5); non-distilled models use more (e.g. 30). |
| `--num_videos` | Number of futures to roll out in parallel from the same context. |
| `--vis_mode` | `none`, `trajectory` (static bird's-eye panel), or `trajectory_ego` (ego-centric panel that follows the current pose). |
| `--speed_scale`, `--yaw_rate_scale` | Global multiplicative factors on the raw speed / yaw-rate conditioning. |
| `--compile` | Wrap the networks with `torch.compile` for faster inference; combine with `--compile_artifacts` to cache the compiled graphs across runs. |

## License (TODO)


## BibTeX (TODO)
```bibtex
@article{orbis2_2026,
  author    = {Mittal, Sudhanshu and Mousakhan, Arian and Galesso, Silvio and
               Farid, Karim and Dienert, Johannes and Sahay, Rajat and Brox, Thomas},
  title     = {Orbis 2: A Hierarchical World Model for Driving},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026},
}
```
