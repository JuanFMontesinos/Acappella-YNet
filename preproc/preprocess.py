import argparse
import os
import sys
from typing import Union, List

import acappella_info

sys.path.append('../..')
import dlib
import cv2
from facenet_pytorch import InceptionResnetV1
import fnmatch
import numpy as np
import json
import imageio
from face_processor import face_processor
import youtube_dl as ydl
from youtube_dl.utils import DownloadError
import glob
import subprocess
import torch
from flerken.video.utils import apply_single
from tqdm import tqdm


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


class YouTubeDownloader(object):

    def __init__(self):
        self.outtmpl = '%(id)s.%(ext)s'
        self.ydl_opts = {
            'format': 'bestvideo+bestaudio',
            'outtmpl': self.outtmpl,
            'logger': None,
            'fragment-retries': 10,
            'retries': 10,
            'abort-on-unavailable-fragment': True,
        }


def download_videos(full_dataset, video_dir_path, dump_unavailable=True):
    print("VIDEO DOWNLOAD IN PROGRESS ... ")
    list_of_removed_videos = []
    ytd = YouTubeDownloader()
    num_rows = len(full_dataset)
    for idx in tqdm(range(num_rows), desc="[# VIDEOS DOWNLOADED]"):
        d = full_dataset[idx]
        video_id = d['ID']
        if glob.glob(os.path.join(video_dir_path, video_id + '*')):
            continue
        if d['Link'] == '' or d['Link'] == "youtube.com/watch?v=":
            continue
        else:
            link = 'http://' + d['Link']
            ytd.ydl_opts['outtmpl'] = os.path.join(video_dir_path, ytd.outtmpl)
            with ydl.YoutubeDL(ytd.ydl_opts) as y:
                try:
                    y.download([link])
                except DownloadError:
                    print('Process failed at video {0}, #{1}'.format(video_id, idx))
                    unwanted_fragments = fnmatch.filter(os.listdir(video_dir_path),
                                                        video_id + '*')
                    for frgmnt in unwanted_fragments:
                        os.remove(os.path.join(video_dir_path, frgmnt))
                    list_of_removed_videos.append(video_id)

                except KeyboardInterrupt:
                    sys.exit()

    if dump_unavailable:
        with open(os.path.join(os.path.dirname(video_dir_path), 'unavailable_videos.json'), 'w') as fp:
            json.dump({'unavailable videos': list_of_removed_videos}, fp, indent=4)
    return list_of_removed_videos


def extract_samples(full_dataset, samples_dir_path, video_download_dir_path, list_of_removed_videos,
                    is_demo_sample=False, FPS=25):
    print("Extracting samples from videos ... ")
    num_rows = len(full_dataset)
    full_length_video_ids = set(os.listdir(video_download_dir_path))
    for idx in tqdm(range(num_rows), desc="# samples extracted", total=num_rows):
        d = full_dataset[idx]
        video_id = d['ID']
        if d['Link'] == '' or d['Link'] == "youtube.com/watch?v=":
            continue
        elif video_id in list_of_removed_videos:
            continue
        else:
            file_name = fnmatch.filter(full_length_video_ids, video_id + '*')[0]
            file_path = os.path.join(video_download_dir_path, file_name)
            init, fin = float(d['Init']), float(d['Fin'])
            begin = int((init // 1) * 60 + (init % 1) * 100)
            end = int((fin // 1) * 60 + (fin % 1) * 100)
            duration = end - begin

            output_file_name = file_name.split('.')[0] + '_' + str.replace(str(init), '.', '_') + '_to_' + str.replace(
                str(fin), '.', '_') + '.mp4'
            if is_demo_sample:
                output_file_dir = os.path.join(samples_dir_path, output_file_name[:-4])
                os.makedirs(output_file_dir, exist_ok=True)
                output_file_path = os.path.join(output_file_dir, output_file_name)
            else:
                output_file_path = os.path.join(samples_dir_path, output_file_name)

            if glob.glob(output_file_path):
                continue
            output_options = ['-ss', str(begin), '-strict', '-2', '-t', str(duration), '-r', str(FPS)]
            result = subprocess.Popen(["ffmpeg", '-i', file_path, *output_options, output_file_path],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout = result.stdout.read()
            stderr = result.stdout.read()
            print(stdout.decode("utf-8"))


def input_face_embeddings(frames: Union[List[str], np.ndarray], resnet: InceptionResnetV1,
                          device: bool, use_half: bool) -> torch.Tensor:
    """
        Get the face embedding

        NOTE: If a face is not detected by the detector,
        instead of throwing an error it zeros the input
        for embedder.

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


def extract_audio(org, dst):
    apply_single(org, dst, output_options=['-ac', '1', '-ar', '16384'], input_options=list(), ext='.wav')


def process_samples(full_dataset, samples_dir_path, splits_dir_path, list_of_removed_videos, device_id, mean_face,
                    FPS=25):
    print("Processing samples ... ")
    num_rows = len(full_dataset)
    splits_dictionary = acappella_info.get_splits_by_subset()
    fp = face_processor(gpu_id=device_id, mean_face=mean_face, ref_img=None, img_size=(244, 224))
    for idx in tqdm(range(num_rows), desc="# samples processed", total=num_rows):
        d = full_dataset[idx]
        video_id = d['ID']
        language = d['Language']
        if language not in ['English', 'Hindi', 'Spanish']:
            language = 'Others'
        gender = d['Gender']
        if d['Link'] == '' or d['Link'] == "youtube.com/watch?v=":
            continue
        elif video_id in list_of_removed_videos:
            continue
        else:
            init, fin = float(d['Init']), float(d['Fin'])
            begin = int((init // 1) * 60 + (init % 1) * 100)
            end = int((fin // 1) * 60 + (fin % 1) * 100)
            duration = end - begin
            num_frames = int(duration * FPS)

            file_name = video_id + '_' + str.replace(str(init), '.', '_') \
                        + '_to_' + str.replace(str(fin), '.', '_') + '.mp4'
            sample_file_path = os.path.join(samples_dir_path, file_name)
            sample_id = file_name[:-4]

            if glob.glob(os.path.join(splits_dir_path, "*", "frames", "*", "*", sample_id + '*')) \
                    and glob.glob(os.path.join(splits_dir_path, "*", "llcp_embed", "*", "*", sample_id + '*')) \
                    and glob.glob(os.path.join(splits_dir_path, "*", "audio", "*", "*", sample_id + '*')) \
                    and glob.glob(os.path.join(splits_dir_path, "*", "landmarks", "*", "*", sample_id + '*')):
                continue

            for key in splits_dictionary.keys():
                if sample_id in splits_dictionary[key]:
                    splits_category = key
                    break
            if splits_category == 'test_unseen':
                splits_category = '_'.join([splits_category, language.lower(), gender.lower()])
            vid = imageio.get_reader(sample_file_path, 'ffmpeg')
            splits_folder_path = os.path.join(splits_dir_path, splits_category)
            frames_path = os.path.join(splits_folder_path, 'frames', language, gender)
            os.makedirs(frames_path, exist_ok=True)
            audio_path = os.path.join(splits_folder_path, 'audio', language, gender)
            os.makedirs(audio_path, exist_ok=True)
            llcp_embed_path = os.path.join(splits_folder_path, 'llcp_embed', language, gender)
            os.makedirs(llcp_embed_path, exist_ok=True)
            landmarks_path = os.path.join(splits_folder_path, 'landmarks', language, gender)
            os.makedirs(landmarks_path, exist_ok=True)
            audio_file_path = os.path.join(audio_path, sample_id + '.wav')
            if not glob.glob(audio_file_path):
                extract_audio(sample_file_path, audio_file_path)

            stacked_frames = np.zeros([num_frames, 128, 96, 3], dtype=np.uint8)
            stacked_llcp_input_frames = np.zeros([num_frames, 160, 160, 3], dtype=np.uint8)
            stacked_landmarks = np.zeros([num_frames, 68, 2], dtype=np.int16)
            good_frame_ids = []
            with tqdm(total=num_frames) as pbar:
                for num_frame in range(num_frames):
                    try:
                        img_raw = vid.get_data(num_frame)
                    except IndexError as e:
                        print("Processing FAILED for sample: " + sample_id)
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

                    img_frame = warped_img[:crop_height, :crop_width, :]
                    llcp_input_frame = cv2.resize(img_frame, (160, 160))
                    stacked_llcp_input_frames[num_frame] = llcp_input_frame
                    img_frame_resized = cv2.resize(img_frame, (96, 128))
                    stacked_frames[num_frame] = img_frame_resized
                    aligned_landmarks_resized = aligned_landmarks * [96 / crop_width, 128 / crop_height]
                    stacked_landmarks[num_frame] = aligned_landmarks_resized
                    pbar.update(1)

                np.save(os.path.join(frames_path, sample_id), stacked_frames)
                np.save(os.path.join(landmarks_path, sample_id), stacked_landmarks)
                del warped_img, landmarks, bbox, vid, stacked_frames, stacked_landmarks

                # extract and dump llcp embeddings
                embed_file_path = os.path.join(llcp_embed_path, sample_id + '.npy')
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

    videos_folder_path = os.path.join(args.data_dir, 'full-length')
    os.makedirs(videos_folder_path, exist_ok=True)
    samples_folder_path = os.path.join(args.data_dir, 'samples')
    os.makedirs(samples_folder_path, exist_ok=True)
    splits_dir_path = os.path.join(args.data_dir, 'splits')
    os.makedirs(splits_dir_path, exist_ok=True)

    mean_face = np.load('mean_face.npy')
    crop_width, crop_height = face_processor.get_width_height(mean_face)
    detector = dlib.get_frontal_face_detector()
    MAX_IMAGE_WIDTH = 1024
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(args.device_id)

    full_dataset = list(acappella_info.csv_gen_full_dataset())

    # downloads full-length videos defined by ids listed in data csv file
    list_of_removed_videos = download_videos(full_dataset, videos_folder_path)
    # Running the previous line again because some video downloads need to be retried
    list_of_removed_videos = download_videos(full_dataset, videos_folder_path)

    """
    with open('/mnt/DATA/dataset/acapsol/acappella/unavailable_videos.json', 'r') as fp:
        list_of_removed_videos = json.load(fp)['unavailable videos']
    """
    # extracts samples from full-length video for given init and fin timestamps
    extract_samples(full_dataset, samples_folder_path, videos_folder_path, list_of_removed_videos)
    # extracts llcp face embeddings and faces from the frames of samples, saves audio
    process_samples(full_dataset, samples_folder_path, splits_dir_path, list_of_removed_videos, args.device_id,
                    mean_face)
