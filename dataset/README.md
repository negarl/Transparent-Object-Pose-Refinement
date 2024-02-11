## Tracebot dataset 

This dataset includes three scenes and was recorded for the project. 
More details about the dataset can be found in Chapter 4 of [Thesis Link](https://drive.google.com/file/d/12WIZiDOofjRkkTLOMlQK83cBeU6k1cLg/view?usp=share_link). 

## Files 
- `camera_d435`: Contains the camera matrix. 
- `groundtruth_handeye.txt`: Contains the camera pose information. 

```
├── dataset
│   └── objects
│       ├── container
│       ├── draintray 
│       └── objects.yaml
└────── scenes 
        └── scene 
            ├── depth  
            ├── masks
            ├── pcl
            ├── rgb
            ├── associations.txt
            └── poses.yaml
```

To get the pose of each object in camera frame `np.linalg.inv(cam_pose) @ obj_pose`. 


 