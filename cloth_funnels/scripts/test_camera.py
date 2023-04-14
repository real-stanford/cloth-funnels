import time
import imageio
from real.setup import *
import os
from tqdm import tqdm

print("Getting top cam")
top_cam = get_top_cam()
# top_cam.get_data()
frames = []
start = time.time()
print("Starting recording...")
total = 2000
pbar = tqdm(total=total)
fps_pbar = tqdm(total=total)
while len(frames) < total:
    top_cam.get_single_data()
    fps_pbar.update()
    if len(frames) and (frames[-1] == top_cam.color_im).all():
        continue
    pbar.update()
    frames.append(top_cam.color_im.copy())
    # frames.append(top_cam.get_rgb())

print(f"Fps:{len(frames)/(time.time() - start)}")

if os.path.exists("test.mp4"):
    os.remove('test.mp4')

# for frame in frames:
path = f'test.mp4'
with imageio.get_writer(path, mode='I', fps=24) as writer:
    for frame in (tqdm(frames)):
        writer.append_data(frame)