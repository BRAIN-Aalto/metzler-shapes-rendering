import argparse
from pathlib import Path
import time

import numpy as np
import wandb

from metzler_renderer.geometry import MetzlerShape
from metzler_renderer.renderer import Object3D, Camera, Renderer


parser = argparse.ArgumentParser()
parser.add_argument(
    "path_to_shapes",
    type=lambda p: Path(p).absolute(),
    help="path to the file with shape strings"
)
parser.add_argument(
    "--project_name",
    default=None,
    type=str,
    help="W&B project name"
)
parser.add_argument(
    "--run_name",
    default=f"run-{time.strftime('%Y-%m-%dT%H:%M:%S')}",
    type=str,
    help="W&B run name"
)
parser.add_argument(
    "--distance_to_camera",
    default=25,
    type=int,
    help="radial distance to the camera from the object"
)
parser.add_argument(
    "--camera_elevation",
    default=-35,
    type=float,
    help="elevation angle (degrees) of the camera position"
)
parser.add_argument(
    "--facecolor",
    default=None,
    type=str,
    help="facecolor of the object"
)
parser.add_argument(
    "--edgecolor",
    default="black",
    type=str,
    help="color of object edges"
)
parser.add_argument(
    "--edgewidth",
    default=1.,
    type=float,
    help="width of object edges"
)
parser.add_argument(
    "--image_size",
    default=[128, 128],
    type=int,
    nargs=2,
    help="size of rendered image"
)
parser.add_argument(
    "--background",
    default="none",
    type=str,
    help="background color of rendered image"
)
parser.add_argument(
    "--dpi",
    default=100,
    type=int,
    help="dpi of rendered image"
)
parser.add_argument(
    "--format",
    default="gif",
    type=str,
    help="video format"
)


args = parser.parse_args()
# start a run
run = wandb.init(
    project=args.project_name,
    name=args.run_name
)
# read shape strings from the file
with open(args.path_to_shapes, "r", encoding="utf-8") as reader:
    shapes = reader.read().splitlines()

# create the camera object
camera = Camera()
# create the renderer object
renderer = Renderer(
    imgsize=tuple(args.image_size),
    dpi=args.dpi,
    bgcolor=args.background,
)

videos = []  # list of wandb video objects

# render 360-degree view for every shape from the file
for shape in shapes:
    # generate 3D object from the shape string
    object3d = Object3D(
        shape=MetzlerShape(shape),
        facecolor=args.facecolor,
        edgecolor=args.edgecolor,
        edgewidth=args.edgewidth
    )
    frames = []  # video frames of object rotation
    for phi in range(360):
        # position the camera
        camera.setSphericalPosition(
            r=args.distance_to_camera,
            theta=args.camera_elevation,
            phi=phi
        )
        # render the image
        renderer.render(object3d, camera)
        # save the image to a numpy array
        # permute dimensions to be channel, height, width
        # add to a frame collection
        frames.append(renderer.save_figure_to_numpy().transpose(2, 0, 1))

    # create a video object and add to a list
    videos.append(
        wandb.Video(
            # numpy tensor must have 4 dimensions (time, channel, height, width)
            np.stack(frames)[None, ...],
            caption=f"SHAPE_{shape.upper()}",
            fps=60,
            format=args.format
        )
    )

# log videos to W&B
wandb.log(
    {
        "Shapes": videos
    }
)
