# Clonetric — Copy-Move Image Forgery Detection (CMFD) Platform

> **Live Deployment:** <https://clonetric-cmfd.streamlit.app/>
>
> **Source Repository:** <https://github.com/patel-samanvay/clonetric-CMFD>

## 1. Executive Overview

**Clonetric** is a multi-engine Digital Image Processing (DIP) and image forensics platform engineered to detect **Copy-Move Image Forgery (CMFD)**—a tampering technique in which a set of pixels is copied from one region of an image and pasted into another region of the same frame to conceal or duplicate objects.

Because cloned regions originate from the same sensor and share identical camera noise, lighting distribution, and color palettes, single-algorithm detectors frequently fail when confronted with scaling, rotation, JPEG re-compression, or smooth low-contrast textures. Clonetric solves this by running **five specialized detection engines in parallel** and fusing their spatial $2\text{D}$ probability maps into a weighted consensus verdict.

## 2. Multi-Engine Detection Architecture

Clonetric evaluates every uploaded or synthesized image across five complementary forensic pipelines coordinated by `orchestrator.py`:

### 2.1 SIFT (Scale-Invariant Feature Transform)

* **Target Anomaly:** Geometrically transformed, scaled, or rotated cloned patches.

* **Methodology:** Extracts scale-space extrema and $128\text{-D}$ gradient orientation descriptors. Performs self-matching using $k\text{-NN}$ ($k=3$), filters ambiguous texture matches via Lowe's Ratio Test, and groups displacement vectors using spatial shift-vector clustering.

* **Filtering Criterion:**
  

  $$
  \frac{d_1}{d_2} < \tau_{\text{ratio}}
  $$

  
  where $d_1$ and $d_2$ represent the Euclidean distances to the first and second nearest non-self descriptors in feature space.

### 2.2 ORB (Oriented FAST and Rotated BRIEF)

* **Target Anomaly:** High-speed structural keypoint duplication.

* **Methodology:** Computes $256\text{-bit}$ binary descriptors paired with Hamming-distance self-matching. Identifies coherent translation vectors $(\Delta x, \Delta y)$ between source and target keypoint clusters:
  

  $$
  \mathbf{v}_i = (x_i' - x_i, \; y_i' - y_i)
  $$

### 2.3 Patch-SIFT (Dense Localized Grid)

* **Target Anomaly:** Smooth or low-texture cloned regions (e.g., sky, walls, grass, water) where sparse keypoint detectors fail to trigger.

* **Methodology:** Samples dense, overlapping spatial blocks across a uniform grid, extracts local descriptors for every patch, and isolates coherent block displacement clusters.

### 2.4 ELA (Error Level Analysis)

* **Target Anomaly:** Differential JPEG quantization artifacts introduced when saving a modified image.

* **Methodology:** Re-encodes the input image in memory at a reference JPEG quality level $Q$ and computes the amplified pixel-wise quantization residual:
  

  $$
  E(x, y) = \alpha \cdot \left\vert{} I_{\text{orig}}(x, y) - I_{Q}(x, y) \right\vert{}
  $$

### 2.5 DL Self-Corr (MobileNetV3 Deep Feature Self-Correlation)

* **Target Anomaly:** Post-processed, blurred, or blended copy-move regions that evade classical pixel/keypoint rules.

* **Methodology:** Passes the image through a truncated pre-trained `MobileNetV3-Small` backbone to extract high-dimensional spatial feature maps $\mathbf{F} \in \mathbb{R}^{C \times H' \times W'}$, followed by all-pairs cosine self-similarity computation outside a local neighborhood exclusion radius:
  

  $$
  S(i, j) = \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\Vert{}\mathbf{f}_i\Vert{}_2 \Vert{}\mathbf{f}_j\Vert{}_2}
  $$

## 3. Consensus Fusion & Dynamic Hyperparameters

The **Ensemble Orchestrator** normalizes the $2\text{D}$ probability masks $M_k(x, y) \in [0, 1]$ produced by all active engines and computes a weighted spatial consensus map:

$$
M_{\text{consensus}}(x, y) = \frac{\sum_{k=1}^{K} w_k \cdot M_k(x, y)}{\sum_{k=1}^{K} w_k}
$$

Users can tune four real-time hyperparameters in the sidebar to control the balance between **False Positives** (natural repeating textures) and **False Negatives** (subtle or post-processed clones):

| Hyperparameter | Range | Forensic Impact | 
| ----- | ----- | ----- | 
| **Ratio Test Threshold (**$\tau_{\text{ratio}}$**)** | $0.50 - 0.95$ | Controls Lowe's Ratio strictness for SIFT and ORB. Lower values reject repetitive background textures; higher values capture degraded or blended clones. | 
| **Minimum Cluster / Match Count** | $3 - 20$ | Sets the minimum number of parallel-shifted vector matches required to flag a region. Lower values detect tiny cloned objects; higher values enforce large structural matches. | 
| **ELA JPEG Quality (**$Q$**)** | $50\% - 98\%$ | Sets the reference re-compression quality for Error Level Analysis. Higher values ($90\% - 95\%$) isolate subtle edits on clean images; lower values help inspect pre-compressed social media images. | 
| **Consensus Threshold** | $0.15 - 0.80$ | Defines the ensemble fusion cutoff score required to transition the executive verdict from `AUTHENTIC` to `TAMPERED`. | 

## 4. Platform Modules & Capabilities

1. **Ensemble Analysis Workspace:**

   * Displays the **Executive Consensus Verdict Banner** (`AUTHENTIC` vs. `TAMPERED`), ensemble confidence percentage, spatial overlap score, and total execution time in milliseconds.

   * Renders side-by-side visual comparisons of the Original Image, the Weighted Consensus Heatmap Overlay, and individual per-engine detection maps.

2. **Cryptographic Chain-of-Custody & EXIF Header:**

   * Automatically computes the file's **SHA-256 cryptographic digest** upon upload.

   * Inspects embedded **EXIF metadata** headers and flags editing software signatures (such as Adobe Photoshop, GIMP, or Canva).

3. **Live Forge & Test Sandbox:**

   * Provides an interactive in-memory forgery generator where users can select source coordinates $(x_s, y_s)$, patch dimensions $(w, h)$, and target destination coordinates $(x_t, y_t)$ to synthesize a copy-move forgery on the fly and immediately benchmark the ensemble's localization accuracy.

4. **CMFD Algorithm Knowledge Base:**

   * Built-in technical documentation detailing the 4-stage processing pipeline, mathematical equations, operational strengths, and failure modes of each engine.

## 5. Repository Structure

```
clonetric-CMFD/
├── app.py                      # Main Streamlit SaaS application & UI design system
├── orchestrator.py             # Multi-engine execution coordinator & heatmap fusion
├── requirements.txt            # Cloud dependencies with CPU-optimized PyTorch wheels
├── .gitignore                  # Git exclusion rules for virtual environments & cache
├── .streamlit/
│   └── config.toml             # Indigo-Slate SaaS dark theme configuration
├── engines/
│   ├── __init__.py             # Engine package initializer
│   ├── sift_engine.py          # SIFT feature extraction & shift-vector clustering
│   ├── orb_engine.py           # ORB binary descriptor matching & shift voting
│   ├── patch_sift_engine.py    # Dense overlapping grid block correlation
│   ├── ela_engine.py           # In-memory JPEG quantization Error Level Analysis
│   └── dl_selfcorr_engine.py   # MobileNetV3 deep feature self-similarity engine
└── Test-Images/                # Curated benchmark images for rapid evaluation

```

## 6. Local Installation & Execution

### Prerequisites

* Python $3.10$ or $3.11$

* `pip` package manager

### Step-by-Step Setup

1. **Clone the Repository:**

   ```
   git clone https://github.com/patel-samanvay/clonetric-CMFD.git
   cd clonetric-CMFD
   
   ```

2. **Create and Activate a Virtual Environment:**

   ```
   python -m venv venv
   
   # Windows (PowerShell / CMD):
   venv\Scripts\activate
   
   # macOS / Linux:
   source venv/bin/activate
   
   ```

3. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   
   ```

4. **Run the Application Locally:**

   ```
   streamlit run app.py
   
   ```

   The application will open automatically in your browser at `http://localhost:8501`.

## 7. Cloud Deployment Notes

Clonetric is deployed on **Streamlit Community Cloud** (<https://clonetric-cmfd.streamlit.app/>) with two key production optimizations:

* **Lightweight CPU PyTorch Wheels:** `requirements.txt` specifies `--extra-index-url https://download.pytorch.org/whl/cpu` to install CPU-only `torch` and `torchvision` binaries, keeping memory usage well below cloud container limits.

* **Resource Caching:** The `MobileNetV3` backbone weights are cached in memory so hyperparameter adjustments re-evaluate swiftly without reloading neural network weights from disk.