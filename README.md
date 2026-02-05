#  Driver Drowsiness

![](<img width="1318" height="657" alt="Image" src="https://github.com/user-attachments/assets/15897f16-a515-4017-8420-74b32d833af3" />)

**Dataset**
https://drive.google.com/drive/folders/18oqffajrF_-P63iH4m_RhoxHvJb4QQ4B?usp=drive_link

**Overview**
TelUP Driver Drowsiness Dataset is an open-source research dataset developed to detect and analyze driver fatigue using visual facial cues under realistic driving conditions. The dataset is designed to support robust drowsiness detection by incorporating temporal facial dynamics, multi-view perspectives, and challenging real-world variations such as occlusions and lighting changes. Under this study, synchronized front-view and side-view facial video data are collected and processed to extract discriminative facial features. Using this dataset, researchers are encouraged to develop and benchmark deep learning models for driver drowsiness classification under diverse and safety-critical conditions.

**Dataset**
The TelUP Driver Drowsiness Dataset is collected in a controlled indoor environment that simulates real driving scenarios. The dataset consists of video recordings from 20 subjects, each performing alert and drowsy driving behaviors captured simultaneously from frontal and side camera views. Each video ranges from 15 to 20 seconds and is recorded at 720p resolution with a frame rate of 30 fps.

From each video, a fixed-length sequence of 32 frames is extracted using uniform temporal sampling. Facial regions are enhanced using CLAHE and cropped before feature extraction. Facial landmarks are obtained using MediaPipe to compute geometric features such as Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose parameters. In parallel, high-level visual embeddings are extracted using the DINOv2-Large model. These geometric and visual features are concatenated to form a unified per-frame representation, which is then organized into temporal sequences.

The dataset includes five subject conditions: no accessory, face mask, white glasses, black glasses, and hat, ensuring robustness against facial occlusions and variable lighting. All sequences are labeled into two classes: drowsiness and non-drowsiness. The data is split using a subject-independent protocol, where subjects 1–15 are used for training and subjects 16–20 are reserved for testing. This dataset supports the evaluation of deep learning architectures for reliable driver drowsiness detection in real-world conditions.