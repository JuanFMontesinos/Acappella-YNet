from __future__ import unicode_literals

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from typing import Union, List

import cv2
import dlib
import imageio
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
from matplotlib.patches import Rectangle
from tqdm import tqdm

from face_processor import face_processor
from preproc.preprocess import extract_audio

sys.path.append('../..')


def get_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def get_progress_points(length, num_points=4, fps=25):
    approx_num_frames = int(length * fps)
    first_point = int(approx_num_frames / (num_points + 1))
    points = np.linspace(first_point, approx_num_frames, num_points + 1)[:-1]
    return points, approx_num_frames


def draw_bounding_boxes(frame, x1, y1, x2, y2):
    plt.imshow(frame)
    rect = Rectangle((x1, y1), (x2 - x1), (y2 - y1), fill=False, color='red')
    plt.axes().add_patch(rect)
    plt.show()


def visualize_frames(video_path, device):
    # FOR NOW, Just to visualize bounding boxes
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    video = mmcv.VideoReader(video_path)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video[:10]]
    # Detect faces
    frame = frames[-1]
    boxes, _ = mtcnn.detect(frame)
    x1, y1, x2, y2 = boxes[0]
    draw_bounding_boxes(frame, x1, y1, x2, y2)


def crop_bbox_zones(sample_path, dest_path, coordinates):
    x1, y1, x2, y2 = coordinates
    subprocess.call(['ffmpeg', '-i', sample_path, '-filter:v',
                     'crop=' + str(x2 - x1) + ':' + str(y2 - y1) + ':' + str(x1) + ':' + str(y1),
                     '-c:a', 'copy', dest_path])


def crop_faces(demos_data, sample_dir_path):
    print("Cropping faces of interest from the video segments... ")
    num_rows = len(demos_data)
    for idx in tqdm(range(num_rows), desc="# samples processed for face cropping", total=num_rows):
        d = demos_data[idx]
        face_ids = d['Face_IDs'].strip('][').split(',')
        if not face_ids == '':
            video_id = d['ID']
            init, fin = float(d['Init']), float(d['Fin'])

            input_file_name = video_id + '_' + str.replace(str(init), '.', '_') + '_to_' + str.replace(
                str(fin), '.', '_') + '.mp4'
            input_file_path = os.path.join(sample_dir_path, input_file_name[:-4], input_file_name)

            face_cropped_coordinates = json.loads(d['Face Crop Region'])
            for idx, face_id in enumerate(face_ids):
                output_path = os.path.join(sample_dir_path, input_file_name[:-4], face_id + '.mp4')
                if not os.path.exists(output_path):
                    crop_bbox_zones(sample_path=input_file_path,
                                    dest_path=output_path,
                                    coordinates=face_cropped_coordinates[idx])


def input_face_embeddings(frames: Union[List[str], np.ndarray], resnet: InceptionResnetV1,
                          device: bool, use_half: bool) -> torch.Tensor:
    """
        Get the face embedding

        NOTE: If a face is not detected by the detector,
        instead of throwing an error it zeros the input
        for embedder.

        NOTE: Memory hungry function, hence the profiler.

        Args:
            frames: Frames from the video
            is_path: Whether to read from filesystem or memory
            resnet: face embedder
            face_embed_cuda: use cuda for model
            use_half: use half precision

        Returns:
            emb: Embedding for all input frames
    """
    # Stack all frames
    frame_tensors = torch.from_numpy(frames)
    del frames
    # Embed all frames
    frame_tensors = frame_tensors.to(device)
    if use_half:
        frame_tensors = frame_tensors.half()

    with torch.no_grad():
        emb = resnet(frame_tensors.permute(0, 3, 1, 2).float())
    if use_half:
        emb = emb.float()
    return emb


def process_samples(samples_dir_path, splits_dir_path, device_id, mean_face, FPS=25):
    print("Processing samples ... ")
    fp = face_processor(gpu_id=device_id, mean_face=mean_face, ref_img=None, img_size=(244, 224))
    examples = sorted(os.listdir(samples_dir_path))
    num_examples = len(examples)

    audio_path = os.path.join(splits_dir_path, 'audio')  # all the face_cropped_samples have the same audio
    os.makedirs(audio_path, exist_ok=True)

    for idx in tqdm(range(num_examples), desc="# demo examples processed", total=num_examples):
        sample_id = examples[idx]

        frames_path = os.path.join(splits_dir_path, 'frames', sample_id)
        os.makedirs(frames_path, exist_ok=True)
        llcp_embed_path = os.path.join(splits_dir_path, 'llcp_embed', sample_id)
        os.makedirs(llcp_embed_path, exist_ok=True)
        landmarks_path = os.path.join(splits_dir_path, 'landmarks', sample_id)
        os.makedirs(landmarks_path, exist_ok=True)

        sample_folder_path = os.path.join(samples_dir_path, sample_id)
        face_cropped_samples = os.listdir(sample_folder_path)
        dest_audio_file_path = os.path.join(audio_path, sample_id + '.wav')
        sample_file_path = os.path.join(sample_folder_path, sample_id + '.mp4')
        num_frames = int(get_duration(sample_file_path) * FPS)
        if not glob.glob(dest_audio_file_path):
            extract_audio(sample_file_path, dest_audio_file_path)

        if len(face_cropped_samples) > 1:
            face_cropped_samples.remove(sample_id + '.mp4')

        for face_cropped_sample in face_cropped_samples:
            face_cropped_sample_name = face_cropped_sample[:-4]
            if glob.glob(os.path.join(splits_dir_path, "frames", sample_id, face_cropped_sample_name + '*')) \
                    and glob.glob(
                os.path.join(splits_dir_path, "llcp_embed", sample_id, face_cropped_sample_name + '*')) \
                    and glob.glob(
                os.path.join(splits_dir_path, "landmarks", sample_id, face_cropped_sample_name + '*')):
                continue
            face_cropped_sample_path = os.path.join(sample_folder_path, face_cropped_sample)
            vid = imageio.get_reader(face_cropped_sample_path, 'ffmpeg')
            stacked_frames = np.zeros([num_frames, 128, 96, 3], dtype=np.uint8)
            stacked_llcp_input_frames = np.zeros([num_frames, 160, 160, 3], dtype=np.uint8)
            stacked_landmarks = np.zeros([num_frames, 68, 2], dtype=np.int16)
            good_frame_ids = []
            with tqdm(total=num_frames) as pbar:
                for num_frame in range(num_frames):
                    try:
                        img_raw = vid.get_data(num_frame)
                    except IndexError as e:
                        print("Processing FAILED for face: " + face_cropped_sample_name + "from " + sample_id)
                        break
                    if img_raw.shape[1] >= MAX_IMAGE_WIDTH:
                        asp_ratio = img_raw.shape[0] / img_raw.shape[1]
                        dim = (MAX_IMAGE_WIDTH, int(MAX_IMAGE_WIDTH * asp_ratio))
                        new_img = cv2.resize(img_raw, dim, interpolation=cv2.INTER_AREA)
                        img = np.asarray(new_img)
                    else:
                        img = img_raw
                    try:
                        warped_img, landmarks, bbox = fp.process_image(img)
                        _, aligned_landmarks, _ = fp.process_image(warped_img)
                        good_frame_ids.append(num_frame)
                    except Exception as e:
                        print("Exception Handled: ", e)
                        continue

                    if warped_img is None:
                        print("NONE TYPE RETURNED")
                        continue

                    img_frame = warped_img[:crop_height, :crop_width, :]  # h=244, w=224
                    llcp_input_frame = cv2.resize(img_frame, (160, 160))
                    stacked_llcp_input_frames[num_frame] = llcp_input_frame
                    img_frame_resized = cv2.resize(img_frame, (96, 128))
                    stacked_frames[num_frame] = img_frame_resized
                    aligned_landmarks_resized = aligned_landmarks * [96 / crop_width, 128 / crop_height]
                    stacked_landmarks[num_frame] = aligned_landmarks_resized
                    pbar.update(1)

                np.save(os.path.join(frames_path, face_cropped_sample_name), stacked_frames)
                np.save(os.path.join(landmarks_path, face_cropped_sample_name), stacked_landmarks)

                del warped_img, landmarks, bbox, vid, stacked_frames, stacked_landmarks

                # extract and dump llcp embeddings
                embed_file_path = os.path.join(llcp_embed_path, face_cropped_sample_name + '.npy')
                num_frames = len(stacked_llcp_input_frames)
                chunk_length = 100
                embeddings = torch.zeros([num_frames, 512], dtype=torch.float32)
                for i in range(int(np.ceil(num_frames / chunk_length))):
                    start = i * chunk_length
                    if start + chunk_length > num_frames:
                        end = num_frames
                    else:
                        end = start + chunk_length
                    embeddings[start:end] = input_face_embeddings(stacked_llcp_input_frames[start:end],
                                                                  resnet=resnet,
                                                                  device=device_id, use_half=False)
                del stacked_llcp_input_frames
                np.save(embed_file_path, embeddings.cpu().numpy())


if __name__ == '__main__':
    # Defining args

    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--data_dir', help='Directory where all dataset files will be stored',
                        type=str, default='/mnt/DATA/dataset/acapsol/acappella')
    parser.add_argument('--device_id', help='Device for training the experiments',
                        type=int, default=0)
    args = parser.parse_args()

    demos_dir_path = os.path.join(args.data_dir, 'demos')
    os.makedirs(demos_dir_path, exist_ok=True)
    videos_folder_path = os.path.join(demos_dir_path, 'full-length')
    os.makedirs(videos_folder_path, exist_ok=True)
    samples_folder_path = os.path.join(demos_dir_path, 'samples')
    os.makedirs(samples_folder_path, exist_ok=True)
    splits_dir_path = os.path.join(demos_dir_path, 'splits')
    os.makedirs(splits_dir_path, exist_ok=True)

    mean_face = np.load('mean_face.npy')
    crop_width, crop_height = face_processor.get_width_height(mean_face)
    detector = dlib.get_frontal_face_detector()
    MAX_IMAGE_WIDTH = 1024
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(args.device_id)

    # \\TODO to add demos.csv to package??
    with open('./demos.csv') as f:
        demos = [{k: v for k, v in row.items()}
                 for row in csv.DictReader(f, skipinitialspace=True)]

    # downloads full-length videos defined by ids listed in data csv file
    # _ = download_videos(demos, videos_folder_path, dump_unavailable=False)

    # extracts samples from full-length video for given init and fin timestamps
    # extract_samples(demos, samples_folder_path, videos_folder_path, [], is_demo_sample=True)

    """
    sample_id = 'SNgnylGkerE_0_15_to_0_2'
    visualize_frames(video_path=os.path.join(samples_folder_path, sample_id, sample_id + '.mp4'), device='cuda:0')
    crop_faces(demos, sample_dir_path=samples_folder_path)
    # The Circle Of Life demo example to be loaded separately without this script.
    """

    # extracts llcp face embeddings and faces from the frames of samples, saves audio
    process_samples(samples_folder_path, splits_dir_path, args.device_id, mean_face)
