"""Reading and writing LeRobot datasets (format v3.0, LeRobot 0.7).

`prepare` copies a recorded LeRobot dataset into a `Store`, resizing every frame once so
training does not decode video. `LeRobotSink` writes one, and is used to put the simulated
demonstrations through the same format the real arm records in.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .data import StoreWriter

IMAGES = "observation.images."


def resize(image, size):
    """(H, W, 3) uint8 -> (size, size, 3) uint8. The whole view is kept; a wide frame is squeezed."""
    if image.shape[:2] == (size, size):
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def open_dataset(source):
    """`source` is a folder holding a LeRobot dataset, or a Hugging Face dataset name."""
    from lerobot.datasets import LeRobotDataset
    if Path(source).exists():
        return LeRobotDataset("local/dataset", root=Path(source), return_uint8=True)
    return LeRobotDataset(source, return_uint8=True)


def prepare(sources, out, img, log=print):
    """Copy one or more LeRobot datasets (same arm, cameras and frame rate) into one store."""
    import torch
    writer, total = None, 0
    for source in sources:
        ds = open_dataset(source)
        meta = ds.meta
        cameras = [k[len(IMAGES):] for k in meta.camera_keys]
        motors = [n.removesuffix(".pos") for n in meta.features["action"]["names"]]
        if writer is None:
            writer = StoreWriter(out, cameras, motors, meta.fps, img)
        elif (cameras, motors, meta.fps) != (writer.meta["cameras"], writer.meta["motors"], writer.meta["fps"]):
            raise ValueError(f"{source} has different cameras, motors or frame rate from the first dataset")
        episodes = [(int(e["dataset_from_index"]), int(e["dataset_to_index"]), e["tasks"][0]) for e in meta.episodes]
        loader = iter(torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False))
        buffer = None
        for n, (a, b, task) in enumerate(episodes):
            state, action, frames = [], [], {c: [] for c in cameras}
            while sum(len(s) for s in state) < b - a:
                batch = buffer if buffer is not None else next(loader)
                buffer = None
                done = sum(len(s) for s in state)
                take = min(b - a - done, len(batch["action"]))
                # The loader is read in order, so check that the frames really are this episode's. A dataset
                # whose indices do not start at 0 or have gaps would otherwise shift every label silently.
                if "index" in batch:
                    got = batch["index"][:take].reshape(-1).tolist()
                    if got != list(range(a + done, a + done + take)):
                        raise ValueError(f"{source}: episode {n} should hold frames {a}..{b - 1}, but the dataset "
                                         f"gave frames {got[0]}..{got[-1]} in its place; its indices are not "
                                         f"contiguous from 0")
                state.append(batch["observation.state"][:take].numpy())
                action.append(batch["action"][:take].numpy())
                for c in cameras:
                    for im in batch[IMAGES + c][:take].permute(0, 2, 3, 1).numpy():
                        frames[c].append(resize(im, img))
                if take < len(batch["action"]):
                    buffer = {k: v[take:] for k, v in batch.items()}
            writer.add_episode(task, np.concatenate(state), np.concatenate(action), {c: np.stack(v) for c, v in frames.items()})
            if (n + 1) % max(1, len(episodes) // 10) == 0:
                log(f"{source}: prepared {n + 1}/{len(episodes)} episodes")
        total += len(episodes)
    if writer is None:
        raise ValueError("no datasets given")
    writer.close()
    return total


class LeRobotSink:
    def __init__(self, root, cameras, motors, fps, img):
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset
        self.cameras = list(cameras)
        names = [f"{m}.pos" for m in motors]
        features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": names},
                    "action": {"dtype": "float32", "shape": (6,), "names": names}}
        for c in self.cameras:
            features[IMAGES + c] = {"dtype": "video", "shape": (img, img, 3), "names": ["height", "width", "channels"]}
        self.ds = LeRobotDataset.create(repo_id="local/dataset", fps=fps, features=features, root=Path(root),
                                        robot_type="so_follower", use_videos=True,
                                        rgb_encoder=RGBEncoderConfig(vcodec="h264"))

    def add_episode(self, task, state, action, frames):
        for t in range(len(state)):
            row = {"observation.state": state[t].astype(np.float32), "action": action[t].astype(np.float32), "task": task}
            for c in self.cameras:
                row[IMAGES + c] = frames[c][t]
            self.ds.add_frame(row)
        self.ds.save_episode(parallel_encoding=False)

    def close(self):
        self.ds.finalize()
