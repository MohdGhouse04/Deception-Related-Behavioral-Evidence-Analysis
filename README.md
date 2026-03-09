# Deception-Related-Behavioral-Evidence-Analysis
Deception analysis has conventionally been based on human judgment, polygraph tests, or psychological assessments, all of which are subjective, intrusive, and inconsistent to varying degrees. This document introduces a multi-phase, reliability-aware system for the detection of deception, related behavioral signs from human motion captured in the video. The main components of the solution are:  interpretable machine learning models with SHAP-based explanations, spatio-temporal motion representation learning with a graph convolutional network trained on expanded skeleton embeddings, out-of-distribution (OOD) validation pipeline. Instead of making a direct deception detection, the method looks at bodily movement patterns and the temporal coordination of the interlocutors under explicitly stated confidence and validity constraints. The system produces understandable confidence, weighted indicators of behavioral evidence and annotated video outputs, which facilitate a cautious and ethical AI-assisted analysis. By focusing on explainability, reliability and validity-aware inference, the proposed system is a tool for human decision, making rather than a source of definite deception judgments.<img width="468" height="127" alt="image" src="https://github.com/user-attachments/assets/0bdcd043-79ca-4b94-92d7-4a5c9f08dba5" />
# methodology 
This study introduces a video-based, three-phase, reliability- aware behavioral evidence analysis system that utilizes skeletal motion representations to examine non-verbal behavior related to deception. The method is specifically aimed at maintaining a balance between interpretability, spatio-temporal learning capacity, and robustness without making overconfident or ethically unsafe claims. The whole pipeline is shown in Fig.1. and Fig.2.
Phase 0: Feature-Based Machine Learning with Explainability

Phase 0 sets up an interpretable baseline for the analysis of deception-related behavior through the use of handcrafted motion features derived from skeleton data.
 
Fig. 1. Dual-Path Training Pipeline for Skeleton-Based Deception Modeling
Skeleton Extraction and Representation
For each video, the frames are extracted, and a pose estimation framework (MediaPipe) is applied to get 33 human skeletal joints for each frame. Each joint is indicated by three spatial coordinates x, y, and z, thus the feature vector for each frame has a dimension of 99. Over time, this leads to a skeleton sequence:

S ∈ R^(T × J × C)                   (1)
Where:
T is the number of frames, 
J joints,
C coordinate channels.
Handcrafted Motion Feature Extraction
On the basis of the skeleton sequences, both the statistical and kinematic features are computed including:
Mean, variance, and range of joint displacements. Joint- velocity and acceleration. Inter-joint distances and posture- dynamics. Short-term temporal window statistics. These features make it possible to interpret the nervousness, rigidity, or abnormal movement variability through motion patterns.
Classification and Explainability
One of the main contributors of deceptive versus truthful conduct is the Random Forest classifier that differentiates the behavior trained on the feature vectors extracted from the data. Random Forest has been selected in this context because of its great resistance to the noise present in the data and its strong capacity to model non-linear correlations.
As a way of ensuring the system is clear and legible to the users, the investigators make use of SHAP (SHapley Additive exPlanations) to get a proper value of the contribution of each motion feature towards the models predictions. This step offers the users of the system understandable behavioral evidence that can be used as a reference and explainability anchor for the rest of the system.
Phase 1: Spatio-Temporal Behavioral Modeling using CTR-GCN
The transition of phase 1 is done from the manually created features to the deep end-to-end spatio-temporal representation that will enable the automatic unfolding of sophisticated joint interaction patterns.
Skeleton Tensor Construction
The original skeleton sequences are converted into graph- compatible tensors:

                     X∈R^{N ×C ×T ×V ×M}                         (2)

Where:
N is the batch size,
C is the coordinate channel dimension,
T is the temporal length,
V denotes the number of joints,
M represents the number of persons (if applicable).


Phase 2: OOD-Aware Reliability Assessment and Confidence Conditioning
In the second phase of the project, the work related to deception aggregation or final decision fusion is not present. The authors only introduce here a reliability gate, which determines behavioral inference validity, uncertainty, or safety.
 
Fig. 2 .  OOD-Aware, Reliability-Gated Video Deception Evidence Analysis Pipeline

Frame-Level Structural Validation
Skeletal joints in each frame undergo scrutiny. If in any frame:
Any of the joint coordinates is NaN or Inf. The joint is close to zero in all spatial dimensions, thus indicating that it was not detected, then the joint is considered as broken. A frame will be invalidated if it features broken joints that account for more than 35% of the total.

Video-level reliability scoring

Let be the total number of frames and the number of invalid frames. The motion confidence score is computed as:
The categories of the input based on this score are as follows:
VALID - visually confirmed skeletal motion.
UNCERTAIN - partially reliable motion.
INVALID - unreliable or non-human input.

  CTR-GCN Architecture
The Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN) is a very powerful tool to model both spatial and temporal dependencies as it is used here:
Spatial graph convolutions learn anatomical joint relationships. Temporal convolutions capture motion evolution across frames. Channel-wise topology refinement dynamically adjusts joint connectivity based on motion context. This part of model simply generalizes on the given dataset (RLDD) and learns the spatio-temporal movements from the skeletons extracted from the user given video.

Confidence - Conditioned Interpretation
In the case of VALID inputs, behavioral evidence of deception is derived from CTR-GCN outputs and motion metrics. If the input falls into the UNCERTAIN category the corresponding explanations are presented with utmost caution. The system is designed in such a way that the deception component is turned off for inputs classified as INVALID. This component is a fail-safe against the generation of deceptive claims on unreliable inputs or non- human activities like animals or corrupted videos.

Phase 2 does not classify deception-merely. Recall that it first assesses the reliability of the input data and then, based on this, decides whether the outputs of the learned models can be trusted.
<img width="468" height="629" alt="image" src="https://github.com/user-attachments/assets/8485478e-e4ea-4192-ac21-e6fd185ddfe2" />
