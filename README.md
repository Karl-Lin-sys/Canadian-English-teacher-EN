# Canadian English Teacher Avatar Project

## Toddler-Level Explanation
Hello there! This is a project to build a "magic puppet that teaches you Canadian English."
- **The Puppet's Face (Avatar)**: The drawing in the computer talks and moves its mouth just like a real person!
- **The Voice Box (Speech Synthesis)**: It produces a beautiful, friendly voice that sounds just like someone from Canada.
- **The Smart Brain (AI)**: When you ask a question in English, it's a clever robot that helps you learn exactly what to say!

## 1. Technical Selection & Architecture
Technical selection was made based on peer-reviewed papers accepted at top-tier conferences (CVPR, ICCV, ECCV, NeurIPS) within the past 3 years, and highly reproducible implementations with top Star counts on GitHub.

### A. Talking Head Generation & Lip Sync
*   **Selected Technology**: Architecture based on **SadTalker (CVPR 2023)** or **EMO (Emote Portrait Alive - 2024)**.
*   **Comparison and Evaluation**:
    *   *MakeItTalk (Older method)*: Directly predicts 2D landmarks from acoustic features, but lacks expressive richness and is vulnerable to class imbalance in training data.
    *   *Wav2Lip (ACM MM 2020)*: Highly generalizable, but possesses a structural defect where the lower half of the face becomes blurry (visual artefacts). *(Note: Canadian spelling used)*
    *   *SadTalker (CVPR 2023)*: Predicts 3DMM (3D Morphable Model) expression coefficients and head pose independently from audio, achieving high-quality rendering. To prevent data leakage, utilizing a pre-trained model solely for inference (frozen) alongside domain adaptation is the optimal approach.
*   **Implementation Framework**: PyTorch (3DMM coefficient predictor implemented via official `torch.nn` modules) and OpenCV (image transformation processing).

### B. Speech Synthesis and Recognition (TTS & ASR)
*   **Selected Technology**: Models based on **VALL-E (NeurIPS 2023)** or **VITS2 (2023)**.
*   **Reasoning**: VITS (ICML 2021) based models suppress overfitting through an end-to-end architecture leveraging Adversarial Learning.

## 2. Structural Defects, Biases, and Solutions
### Root Causes
1.  **Selection Bias:** When the training data for ASR (Automatic Speech Recognition) or lip-sync models (e.g., VoxCeleb, HDTF) is heavily skewed towards specific ethnic groups or native English speakers, the recognition rate drops significantly for non-native inputs (like an L2 learner or your friend from Venezuela), causing the lip-sync to break down or exhibit biased behaviour.
2.  **Structural Defect:** Traditional deterministic regression models (using MSE Loss) fail to learn the "one-to-many problem" (the fact that there are multiple correct facial expressions for the same audio). As a result, they suffer from regression to the mean, consistently generating an average (blurry or expressionless) face.

### Three Theoretically Backed Solutions
1.  **Adopting General Features via Self-Supervised Learning (SSL) (Insights from NeurIPS 2023, etc.):**
    *   Replace legacy MFCC features with HuBERT (Hidden-Unit BERT), a speech feature extractor pre-trained on diverse, unlabelled, multilingual data. Recent research proves that SSL representations are less susceptible to specific language phoneme biases and perform robustly even on non-native speech.
2.  **Eliminating Bias via Domain-Adversarial Training:**
    *   Incorporate a Gradient Reversal Layer (GRL) into the learning network to train the feature extractor to fool a Discriminator attempting to classify the input audio as "native" or "non-native." This secures a pure linguistic feature space independent of accent (domain), mitigating performance degradation caused by dataset skew (class imbalance).
3.  **Ensuring Diversity via Probabilistic Modelling (Diffusion Models/VAE) (CVPR 2023/2024):**
    *   Discard deterministic loss functions (like MSE) and transition to an approach that samples expression coefficients from a probability distribution using Diffusion Models or Conditional VAEs. This prevents overfitting while generating natural and diverse avatar movements—such as subtle blinks and head swaying—for the exact same audio.

## 3. Time/Space Complexity (Big-O) & Python Code Implementation
The source code for this project is written in `avatar_model.py`. To prevent memory leaks and unnecessary object creation, we adhere to the best practices outlined in the official documentation for PyTorch and OpenCV.
