# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os


os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import colorsys
import io
import pickle
import shutil
import time

import cv2
import joblib
import numpy as np
import torch

from multi_person_tracker import MPT
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.data_utils.kp_utils import convert_kps
from lib.dataset.inference import Inference
from lib.models.vibe import VIBE_Demo
from lib.utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    download_ckpt,
    download_youtube_clip,
    images_to_video,
    prepare_rendering_results,
    smplify_runner,
    video_to_images,
)
from lib.utils.pose_tracker import run_posetracker
from lib.utils.renderer import Renderer
from lib.utils.smooth_pose import smooth_pose
from zipreader import ZipReader


MIN_NUM_FRAMES = 25


def load_data(args):

    split_meta_file = os.path.join(args.data_folder, f"{args.split}.pkl")
    with open(split_meta_file, "rb") as f:
        split_meta_dicts = pickle.load(f)
    # keys: keys(['seq_len', 'img_dir', 'name', 'video_file', 'label'])
    return split_meta_dicts


def read_img(frame_path):
    img_data = ZipReader.read(frame_path)
    rgb_im = Image.open(io.BytesIO(img_data)).convert("RGB")
    return rgb_im


def prepare_image_folder(args, vid_dict):
    video_name = vid_dict["name"]
    seq_len = vid_dict["seq_len"]
    img_dir = vid_dict["img_dir"]
    frames_zip_file = args.frames_zip_file

    image_folder = os.path.join("/tmp/MSASL", video_name)
    os.makedirs(image_folder, exist_ok=True)
    frame_paths = [f"{frames_zip_file}@{img_dir}{frame_id:04d}.png" for frame_id in range(seq_len)]

    frame_images = [read_img(frame_path) for frame_path in frame_paths]
    for frame_id, frame_image in enumerate(frame_images):
        frame_image.save(os.path.join(image_folder, f"{frame_id:04d}.png"))
    orig_width, orig_height = frame_images[0].size
    return video_name, image_folder, seq_len, orig_width, orig_height


def process_one_video(args, mot, model, video_name, image_folder, num_frames, orig_width, orig_height, device):
    bbox_scale = 1.1

    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]["frames"].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    assert (
        len(list(tracking_results.keys())) == 1
    ), f"No person or more than one person detected in the video {video_name}"

    person_id = list(tracking_results.keys())[0]
    bboxes = joints2d = None

    if args.tracking_method == "bbox":
        bboxes = tracking_results[person_id]["bbox"]
    elif args.tracking_method == "pose":
        joints2d = tracking_results[person_id]["joints2d"]

    frames = tracking_results[person_id]["frames"]

    dataset = Inference(
        image_folder=image_folder,
        frames=frames,
        bboxes=bboxes,
        joints2d=joints2d,
        scale=bbox_scale,
    )

    bboxes = dataset.bboxes
    frames = dataset.frames
    has_keypoints = True if joints2d is not None else False

    dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

    with torch.no_grad():

        pred_cam, pred_verts, pred_pose, pred_betas = [], [], [], []
        pred_joints3d, smpl_joints2d, norm_joints2d = [], [], []

        for batch in dataloader:
            if has_keypoints:
                batch, nj2d = batch
                norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

            batch = batch.unsqueeze(0)
            batch = batch.to(device)

            batch_size, seqlen = batch.shape[:2]
            output = model(batch)[-1]

            pred_cam.append(output["theta"][:, :, :3].reshape(batch_size * seqlen, -1))
            pred_verts.append(output["verts"].reshape(batch_size * seqlen, -1, 3))
            pred_pose.append(output["theta"][:, :, 3:75].reshape(batch_size * seqlen, -1))
            pred_betas.append(output["theta"][:, :, 75:].reshape(batch_size * seqlen, -1))
            pred_joints3d.append(output["kp_3d"].reshape(batch_size * seqlen, -1, 3))
            smpl_joints2d.append(output["kp_2d"].reshape(batch_size * seqlen, -1, 2))

        pred_cam = torch.cat(pred_cam, dim=0)
        pred_verts = torch.cat(pred_verts, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)
        smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
        del batch

    # ========= Save results to a pickle file ========= #
    pred_cam = pred_cam.cpu().numpy()
    pred_verts = pred_verts.cpu().numpy()
    pred_pose = pred_pose.cpu().numpy()
    pred_betas = pred_betas.cpu().numpy()
    pred_joints3d = pred_joints3d.cpu().numpy()
    smpl_joints2d = smpl_joints2d.cpu().numpy()

    # Runs 1 Euro Filter to smooth out the results
    if args.smooth:
        min_cutoff = args.smooth_min_cutoff  # 0.004
        beta = args.smooth_beta  # 1.5
        # print(f"Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}")
        pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas, min_cutoff=min_cutoff, beta=beta)

    orig_cam = convert_crop_cam_to_orig_img(cam=pred_cam, bbox=bboxes, img_width=orig_width, img_height=orig_height)

    joints2d_img_coord = convert_crop_coords_to_orig_img(
        bbox=bboxes,
        keypoints=smpl_joints2d,
        crop_size=224,
    )

    output_dict = {
        "pred_cam": pred_cam,
        "orig_cam": orig_cam,
        "verts": pred_verts,
        "pose": pred_pose,
        "betas": pred_betas,
        "joints3d": pred_joints3d,
        "joints2d": joints2d,
        "joints2d_img_coord": joints2d_img_coord,
        "bboxes": bboxes,
        "frame_ids": frames,
    }

    video_output_path = os.path.join(args.output_folder, "vibe_outs", f"{video_name}.pkl")
    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

    joblib.dump(output_dict, video_output_path)

    vibe_results = {person_id: output_dict}
    if not args.no_render:
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

        output_img_folder = f"{image_folder}_output"
        os.makedirs(output_img_folder, exist_ok=True)

        # print(f"Rendering output video, writing frames to {output_img_folder}")

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted(
            [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith(".png") or x.endswith(".jpg")
            ]
        )

        for frame_idx in range(len(image_file_names)):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            if args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data["verts"]
                frame_cam = person_data["cam"]

                mc = mesh_color[person_id]

                mesh_filename = None

                if args.save_obj:
                    mesh_folder = os.path.join(args.output_folder, "meshes", f"{person_id:04d}")
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f"{frame_idx:06d}.obj")

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                if args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        angle=270,
                        axis=[0, 1, 0],
                    )

            if args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f"{frame_idx:06d}.png"), img)

            if args.display:
                cv2.imshow("Video", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        save_name = f'{video_name}.mp4'
        save_name = os.path.join(args.output_folder, "vibe_videos", save_name)
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        # print(f"Saving result video to {save_name}")
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

    return output_dict


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    split_meta_dicts = load_data(args)

    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format="dict",
        yolo_img_size=args.yolo_img_size,
    )

    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)
    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt["gen_state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from "{pretrained_file}"')

    all_vibe_outputs = []
    for vid_dict in tqdm(split_meta_dicts):
        video_name, image_folder, num_frames, orig_width, orig_height = prepare_image_folder(args, vid_dict)
        vibe_out = process_one_video(
            args, mot, model, video_name, image_folder, num_frames, orig_width, orig_height, device
        )
        all_vibe_outputs.append(vibe_out)

        shutil.rmtree(image_folder)

    video_all_path = os.path.join(args.output_folder, f"vibe_msasl_all.pkl")

    joblib.dump(all_vibe_outputs, video_all_path)
    # print("================= END =================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vid_file", type=str, help="input video path or youtube link")
    parser.add_argument("--output_folder", type=str, help="output folder to write results")
    parser.add_argument(
        "--tracking_method",
        type=str,
        default="bbox",
        choices=["bbox", "pose"],
        help="tracking method to calculate the tracklet of a subject from the input video",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo",
        choices=["yolo", "maskrcnn"],
        help="object detector to be used for bbox tracking",
    )
    parser.add_argument("--yolo_img_size", type=int, default=416, help="input image size for yolo detector")
    parser.add_argument(
        "--tracker_batch_size", type=int, default=12, help="batch size of object detector used for bbox tracking"
    )
    parser.add_argument(
        "--staf_dir",
        type=str,
        default="/home/mkocabas/developments/openposetrack",
        help="path to directory STAF pose tracking method installed.",
    )
    parser.add_argument("--vibe_batch_size", type=int, default=450, help="batch size of VIBE")
    parser.add_argument("--display", action="store_true", help="visualize the results of each step during demo")
    parser.add_argument(
        "--run_smplify",
        action="store_true",
        help="run smplify for refining the results, you need pose tracking to enable it",
    )
    parser.add_argument("--no_render", action="store_true", help="disable final rendering of output video.")
    parser.add_argument("--wireframe", action="store_true", help="render all meshes as wireframes.")
    parser.add_argument("--sideview", action="store_true", help="render meshes from alternate viewpoint.")
    parser.add_argument("--save_obj", action="store_true", help="save results as .obj files.")
    parser.add_argument("--smooth", action="store_true", help="smooth the results to prevent jitter")
    parser.add_argument(
        "--smooth_min_cutoff",
        type=float,
        default=0.004,
        help="one euro filter min cutoff. " "Decreasing the minimum cutoff frequency decreases slow speed jitter",
    )
    parser.add_argument(
        "--smooth_beta",
        type=float,
        default=0.7,
        help="one euro filter beta. " "Increasing the speed coefficient(beta) decreases speed lag.",
    )

    parser.add_argument(
        "--frames_zip_file",
        type=str,
        default="/D_data/SL/data/MSASL/msasl_frames1.zip",
        help="input frames zip file path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="split of the dataset to run the demo on.",
    )
    parser.add_argument("--data_folder", type=str, default="/D_data/SL/data/MSASL/")

    args = parser.parse_args()
    main(args)
