# wf-3d-pose-tracking

Classes and methods for performing 3D pose tracking from 2D poses:

* Functions which load 2D pose data from Wildflower S3 directories

* Functions which perform very basic 3D pose reconstruction from multi-camera 2D pose data

* Functions which support basic visualization of 2D and 3D pose data

## Testing

To run tests, change working directory to `/tests` and run one of the following.

`python 3d_reconstruction_example.py` pulls a single frame of 2D pose data from S3 and performs a simple 3D pose reconstruction

`python 3d_pose_tracking_example.py` pulls multiple frames of 2D pose data from a local file and builds a set of 3D pose tracks

These scripts pull camera calibration parameter files and sample 2D pose data from `tests/data`

To run these scripts (and to use any of the functions data from S3), you need to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables to keys belonging to an account that is a member of the Wildflower `cameras` group.
