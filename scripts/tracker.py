import argparse
from pyservicemaker import Pipeline, Flow

parser = argparse.ArgumentParser(description="Capstone Tracker: Multi-Stream vs. Local Cam")
parser.add_argument("--use_cam", action="store_true", help="Use laptop use_cam instead of video files")
args = parser.parse_args()

pipeline = Pipeline("multi_camera_detector")

# infer_config = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"
infer_config = "/workspace/models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt"

if args.use_cam:
    sources = ["v4l2:///dev/video0"]
    print("LOG: Using local camera")
else:
    base_path = "/workspace/sample_video/20250828132109/"
    sources = [
        f"{base_path}D01_20250828132109.mp4",
        f"{base_path}D02_20250828132109.mp4",
        f"{base_path}D03_20250828132109.mp4",
        f"{base_path}D04_20250828132109.mp4"
    ]
    print(f"LOG: Using {len(sources)} video streams at {base_path}")

Flow(pipeline).batch_capture(sources).infer(infer_config).render()()
