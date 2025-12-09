# Eye Visibility + Landmark Multi-Task Model

## Overview
This project trains a multi-task CNN model with:
- A binary classifier head predicting eye visibility.
- A 6-landmark regression head (12 values: x1,y1,...,x6,y6) for visible eyes.
Landmark loss is masked when the eye is not visible.

## Data
Training CSV columns:
`video_id,frame_key,eye_side,eye_visibility,eye_crop_path,eye_bbox_face,landmarks_cordinates_inside_eye_bbx`

Landmarks are pixel coordinates relative to the *eye crop*. If visibility=0, landmarks list may be empty `[]`.

Place your CSVs:
```
data/train.csv
data/val.csv
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Edit `configs/default.yaml` to adjust paths, hyperparameters, and augmentations.

## Training
```bash
python -m src.train --config configs/default.yaml
```

Resumes automatically if `checkpoint.resume: true` and a checkpoint exists.

## Inference
```bash
python -m src.inference \
  --checkpoint outputs/best_model.pt \
  --csv data/val.csv \
  --output_dir outputs/inference_samples
```

## Visualize Samples
```bash
python -m src.visualize_samples \
  --checkpoint outputs/best_model.pt \
  --config configs/default.yaml \
  --csv data/val-v2.clean.csv \
  --output_dir outputs/vis_val \
  --limit 40 \
  --show_gt
```

## Metrics
Classification: accuracy, precision, recall, F1.
Landmarks (visible only): MSE per point, Normalized Mean Error (NME against crop diagonal), PCK@threshold.

## Extensibility
- Switch backbone in `configs/default.yaml`.
- Enable horizontal flip once landmark ordering semantics confirmed.
- Replace coordinate regression with heatmaps (future).
- Add temporal smoothing outside this repo.

## Notes
Ensure all `eye_crop_path` files exist. Missing files are logged and skipped.



# Generation of training data from labelling data and labelled xml files

1.Refer to "/inwdata2a/sudhanshu/video-path-creation/extraction-rule.txt" document to see the extraction process

2. Now use "/inwdata2a/sudhanshu/nd_data_processing_scripts/gaze_estimation_mydata.py" gaze_estimation_mydata.py contains the main pipeline that generates face and eye crops json files for all frames of each folder.
it also gives you options in switching bt function generate_face_crop which saves all face-&-eye crops and generate_face_crop1 which only saves random samples of face-&-eye crops
it takes txt file of video folder names to consider only those folders for generating the face and eye crops
when you run this script, it will save face_detections.json and gaze_eye_crops-final.json in each folder inside output_root



3. once we have face_detections.json file now use "/inwdata2a/sudhanshu/video-path-creation/extract-face-crops.py"
 This takes the full video frames, and face_detections.json file folder and saves the cropped faces in new folder


4. Those cropped faces goes for labelling!



5. Now that we have cropped faces and the corresponding labelled xml file run "/inwdata2a/sudhanshu/eye_multitask_training/scripts/generate-eye-landmark-csv.py"

6. "generate-eye-landmark-csv.py" takes xml file, the cropped images folder and saves the csv file named (eye_landmarks.csv) with below information
     video_id,frame_key,eye_side,eye_visibility,path_to_dataset,eye_bbox_face,landmarks_coordinates_inside_eye_bbox

7. The eyecrops are created using the boundry which includes landmarks + padding and making them square 

8. (just for confirmation) Run "/inwdata2a/sudhanshu/eye_multitask_training/scripts/generate-eye-crops-overlay.py" which only takes the saved csv file (in point 6) and saves the overlays

9. now use "/inwdata2a/sudhanshu/eye_multitask_training/scripts/split_train_val.py" to split the csv file into train.csv and val.csv in 80:20 ratio

10. Now that we are confirmed that our final training csv file (generated in point 6) we are good to go for tarining!



--------------------------------------------------------------Training Instructions----------------------------------------------------------------











