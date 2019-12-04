import pose_tracking_3d
import cv_utils
import numpy as np
import json


classroom_name = 'sandbox'
datetime = np.datetime64('2018-07-04T18:23:00')
camera_names = ['camera01', 'camera02', 'camera03', 'camera04']
data_directory = './data'

camera_calibration_parameters = cv_utils.fetch_camera_calibration_data_from_local_drive_multiple_cameras(
    camera_names = camera_names,
    camera_calibration_data_directory = './data')

all_2d_poses = 3d_pose_tracking.Poses2D.from_openpose_timestep_wildflower_s3(
        classroom_name,
        camera_names,
        datetime)

all_3d_poses = 3d_pose_tracking.Pose3DGraph.from_poses_2d_timestep(
    all_2d_poses,
    camera_calibration_parameters)

matched_3d_poses = all_3d_poses.extract_matched_poses()

print('\nMatched pose keypoints:')
print(matched_3d_poses.keypoints())

print('\nMatched pose projection errors:')
print(matched_3d_poses.projection_errors())

print('\nMatch indices:')
print(matched_3d_poses.pose_indices())
