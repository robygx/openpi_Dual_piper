# DualPiper Dataset Configuration

This directory contains configuration for fine-tuning openpi models on DualPiper dual-arm robot data.

## DualPiper Specifications

- **Robot Type**: Dual-arm robot (similar to ALOHA)
- **State Dimension**: 14 (left arm 6 joints + 1 gripper, right arm 6 joints + 1 gripper)
- **Action Dimension**: 14 (same as state)
- **Cameras**: 3 cameras
  - `cam_high`: High-angle third-person view
  - `cam_left_wrist`: Left wrist camera
  - `cam_right_wrist`: Right wrist camera
  - Note: No `cam_low` (unlike ALOHA which has 4 cameras)
- **Action Type**: Absolute positions (converted to delta during training)
- **FPS**: 50 (adjustable)

## Key Differences from ALOHA

| Feature | ALOHA | DualPiper |
|---------|-------|-----------|
| Cameras | 4 (high, low, left_wrist, right_wrist) | 3 (high, left_wrist, right_wrist) |
| State/Action Dim | 14 | 14 |
| Delta Actions | Yes | Yes |
| Camera Resolution | Varies | Adjust to your setup |

## Files

- `convert_dual_piper_data_to_lerobot.py`: Template script to convert raw DualPiper data to LeRobot format
- Configuration in `/home/ygx/VLA/openpi/src/openpi/training/config.py`:
  - `LeRobotDualPiperDataConfig`: Data configuration class
  - `pi05_dual_piper`: Training configuration
- Policy in `/home/ygx/VLA/openpi/src/openpi/policies/dual_piper_policy.py`:
  - `DualPiperInputs`: Input transformation
  - `DualPiperOutputs`: Output transformation

## Usage

### 1. Convert Your Data

Modify and run the conversion script:

```bash
cd /home/ygx/VLA/openpi

uv run examples/dual_piper/convert_dual_piper_data_to_lerobot.py \
    --raw-dir /path/to/your/raw/data \
    --repo-id your_username/dual_piper
```

**Important**: You need to edit `convert_dual_piper_data_to_lerobot.py` to load your specific data format. The current script is a template.

### 2. Update Config

Edit `/home/ygx/VLA/openpi/src/openpi/training/config.py` and update line 902:

```python
repo_id="your_username/dual_piper",  # Change to your actual repo_id
```

### 3. Compute Normalization Stats

```bash
cd /home/ygx/VLA/openpi

uv run scripts/compute_norm_stats.py --config-name pi05_dual_piper
```

This will create `assets/your_username/dual_piper/norm_stats.json`.

### 4. Train

```bash
cd /home/ygx/VLA/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_dual_piper \
    --exp-name=dual_piper_finetune \
    --overwrite
```

Training checkpoints will be saved to `checkpoints/pi05_dual_piper/dual_piper_finetune/`.

### 5. Inference

Start the policy server:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_dual_piper \
    --policy.dir=checkpoints/pi05_dual_piper/dual_piper_finetune/30000
```

Then run your DualPiper inference script.

## Data Format Requirements

Your data should contain:

### Observations per timestep
- `observation.state`: (14,) float32 - joint positions and gripper states
- `observation.images.cam_high`: (H, W, 3) uint8 - high camera image
- `observation.images.cam_left_wrist`: (H, W, 3) uint8 - left wrist camera image
- `observation.images.cam_right_wrist`: (H, W, 3) uint8 - right wrist camera image

### Actions per timestep
- `action`: (14,) float32 - target joint positions and gripper commands

### Tasks
- `task`: str - language instruction describing the task

## Action Space

The 14-dimensional action space represents:
- **Dimensions 0-5**: Left arm joint angles (absolute)
- **Dimension 6**: Left gripper (absolute, 0=closed, 1=open)
- **Dimensions 7-12**: Right arm joint angles (absolute)
- **Dimension 13**: Right gripper (absolute, 0=closed, 1=open)

During training, joint angles (dims 0-5, 7-12) are automatically converted to delta actions, while gripper actions (dims 6, 13) remain absolute.

## Troubleshooting

### Import Error
If you get an import error for `dual_piper_policy`, make sure you've installed the package:
```bash
cd /home/ygx/VLA/openpi
uv pip install -e .
```

### Config Not Found
If `pi05_dual_piper` config is not found, check that:
1. The import statement was added to `config.py` (line 22)
2. `LeRobotDualPiperDataConfig` class is defined (line 360)
3. `TrainConfig` is added to `_CONFIGS` list (line 894)

### Data Loading Issues
If data loading fails, check:
1. Your repo_id matches the actual HuggingFace dataset
2. The features in your data match the expected format
3. Images are in the correct format (uint8, HWC or CHW)

## References

- ALOHA example: `/home/ygx/VLA/openpi/examples/aloha_real/`
- LIBERO example: `/home/ygx/VLA/openpi/examples/libero/`
- LeRobot documentation: https://github.com/huggingface/lerobot
