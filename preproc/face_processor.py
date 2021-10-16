import sys
sys.path.append(sys.path[0] + '/..')
from face_align import FaceAlignment, LandmarksType
import numpy as np
import skvideo.io
import cv2
import csv
from skimage import transform as tf
import os

# Code obtained from https://github.com/DinoMan/face-processor

stablePntsIDs = [33, 36, 39, 42, 45]


class face_processor():
    def __init__(self, mean_face=None, ref_img=None, cuda=True, img_size=None, gpu_id=0):
        device = 'cuda:'+str(gpu_id)
        self.fa = FaceAlignment(LandmarksType._2D, device=device, flip_input=False)
        self.ref_img = ref_img
        self.mean_face = None
        self.img_size = img_size
        if self.ref_img is None:
            if mean_face is not None:
                if isinstance(mean_face, str):
                    self.mean_face = np.load(mean_face)
                else:
                    self.mean_face = mean_face
            else:
                self.mean_face = np.load(os.path.split(__file__)[0] + "/data/mean_face.npy")

    def get_transform(self, image):
        try:
            landmarks = self.fa.get_landmarks(image)[0]
        except Exception as ex:
            print(ex)
            return None

        stable_points = landmarks[stablePntsIDs, :]

        if self.mean_face is None and self.ref_img is not None:
            self.mean_face = self.fa.get_landmarks(self.ref_img)[0]

        warped_img, trans = self.warp_img(stable_points,
                                          self.mean_face[stablePntsIDs, :],
                                          image)
        return trans

    def process_image(self, image):
        try:
            landmarkset = self.fa.get_landmarks(image)
            landmarks = landmarkset[0]
            bbox = landmarkset[1]
        except:
            return None

        stable_points = landmarks[stablePntsIDs, :]

        if self.mean_face is None and self.ref_img is not None:
            self.mean_face = self.fa.get_landmarks(self.ref_img)[0]

        warped_img, trans = self.warp_img(stable_points,
                                          self.mean_face[stablePntsIDs, :],
                                          image, output_shape=self.img_size)
        return warped_img, landmarks, bbox

    def normalise_face(self, video_input, landmarks_input=None, window_size=7):

        if self.mean_face is None and self.ref_img is not None:
            self.mean_face = self.fa.get_landmarks(self.ref_img)[0]

        # Check if we should read from file
        if isinstance(video_input, str):
            video = skvideo.io.vread(video_input)
        else:
            video = video_input

        if window_size % 2 == 0:
            window_size += 1

        # If we have landmarks
        if landmarks_input is not None:
            if isinstance(landmarks_input, str):
                landmarks = self.parse_landmarks_file(landmarks_input)
            else:
                landmarks = landmarks_input
        else:
            landmarks = []

        if video.shape[0] < window_size or (landmarks_input is not None and len(landmarks) == 0):
            return None

        trans = None
        projected_landmarks = []
        out_vid_size = list(video.shape)
        if self.img_size is not None:
            out_vid_size[1] = self.img_size[0]
            out_vid_size[2] = self.img_size[1]

        out_video = np.empty(out_vid_size)
        for frame_no in range(0, video.shape[0]):
            if frame_no + window_size <= video.shape[0]:
                avg_stable_points = np.zeros([len(stablePntsIDs), 2])
                for i in range(0, window_size):
                    if landmarks_input is None:
                        landmarks.append(self.fa.get_landmarks(video[frame_no + i])[0])
                        avg_stable_points += landmarks[-1][stablePntsIDs, :]
                    else:
                        avg_stable_points += landmarks[frame_no + i][stablePntsIDs, :]

                avg_stable_points /= window_size
                out_video[frame_no], trans = self.warp_img(avg_stable_points,
                                                           self.mean_face[stablePntsIDs, :],
                                                           video[frame_no], output_shape=self.img_size)
            else:
                if landmarks_input is None:
                    landmarks.append(self.fa.get_landmarks(video[frame_no])[0])

                out_video[frame_no] = self.apply_transform(trans, video[frame_no], output_shape=self.img_size)

            projected_landmarks.append(trans(landmarks[frame_no]))

        return out_video, projected_landmarks

    @staticmethod
    def apply_transform(transform, img, output_shape=None):
        warped = tf.warp(img, inverse_map=transform.inverse, output_shape=output_shape)
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped

    @staticmethod
    def warp_img(src, dst, img, output_shape=None):
        tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=output_shape)  # wrap the frame image
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped, tform

    @staticmethod
    def draw_points(image, points, tag=True, in_place=False, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if in_place:
            img = image
        else:
            img = np.copy(image)

        for i in range(points.shape[0]):
            if tag:
                cv2.putText(img, str(i), (int(points[i, 0]), int(points[i, 1])), font, 0.23, color)
            else:
                cv2.circle(img, (int(points[i, 0]), int(points[i, 1])), 1, color)

        return img

    @staticmethod
    def get_width_height(points):
        tl_corner, br_corner = face_processor.find_corners(points)

        width = (br_corner - tl_corner)[0]
        height = (br_corner - tl_corner)[1]

        return width, height

    @staticmethod
    def find_corners(points):
        tl = np.array([points[:, 0].min(), points[:, 1].min()])
        br = np.array([points[:, 0].max(), points[:, 1].max()])

        return tl.astype(int), br.astype(int)

    @staticmethod
    def get_frame_rate(video_file):
        return skvideo.io.ffprobe(video_file)["video"]["@r_frame_rate"]

    @staticmethod
    def parse_landmarks_file(landmarks_file):
        video_landmarks = []
        with open(landmarks_file, 'rt', encoding="ascii") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')

            for frame_no, landmarks in enumerate(csvreader):
                frame_landmarks = np.zeros([68, 2])
                for point in range(1, len(landmarks), 2):
                    frame_landmarks[point // 2, 0] = int(landmarks[point + 1])
                    frame_landmarks[point // 2, 1] = int(landmarks[point])

                    if int(landmarks[point]) == -1:
                        return []
                video_landmarks.append(frame_landmarks)

        return video_landmarks, landmarks

    @staticmethod
    def offset_mean_face(mean_landmarks, offset_percentage=[0, 0]):
        tl_corner, br_corner = face_processor.find_corners(mean_landmarks)

        width = (br_corner - tl_corner)[0]
        height = (br_corner - tl_corner)[1]

        offset = np.array([int(offset_percentage[0] * width), int(offset_percentage[1] * height)])
        return mean_landmarks - tl_corner + offset

    def find_mean_face(self, files, offset_percentage=[0, 0]):
        mean_landmarks = np.zeros([68, 2])
        number_of_faces = 0
        for video_file in files:
            video = skvideo.io.vread(video_file)
            for frame_no in range(0, video.shape[0], 10):
                number_of_faces += 1
                mean_landmarks += self.fa.get_landmarks(video[frame_no])[0]

            if number_of_faces == 1000:
                break

        mean_face = np.multiply(1 / number_of_faces, mean_landmarks)

        self.mean_face = self.offset_mean_face(mean_face, offset_percentage)

        return self.mean_face