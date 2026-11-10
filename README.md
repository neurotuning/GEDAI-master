## GEDAI denoising toolbox for Matlab (EEGLAB/Brainstorm/FieldTrip)


<img width="150" height="135" alt="GEDAI-CIBM_logo" src="https://github.com/user-attachments/assets/bc1775f6-c3c7-4273-a2ac-6ac855653039" />


## Generalized Eigenvalue De-Artifacting Instrument (GEDAI) 
**A)** EEG may be considered to be a mixture of electrical activities from a brain “signal” (sub)space, and one containing different types of non-cerebral noise or “artifacts”. This mixture may be “unmixed” by linear decomposition techniques (e.g. PCA or ICA) into separate "components" with individual source locations and respective time-courses. However, although PCA and ICA leverage statistical properties within mixed data to recover underlying sources, they are "blind" source separation methods, functioning without a priori knowledge of the original signals or their mixing process. GEDAI combines theoretical knowledge of the brain’s “signal” subspace with generalized eigenvalue decomposition (GEVD) to automatically separate brain and artifact components. Here, a theoretical model of EEG generation is used as an estimate of the brain’s “noise-free” subspace.


![F1 large](https://github.com/user-attachments/assets/2cd836e0-1b22-4b64-8766-026ca1195633)


**B)** Each data covariance matrix (dataCOV) is decomposed into source components with GEVD.  
**C)** The GEVD uses a fixed theoretical reference matrix (refCOV) across all epochs, based on the leadfield matrix of an EEG forward model.  
**D)** To determine the optimal threshold separating brain and artifactual subspaces, output EEG data is evaluated using the Signal & Noise Subspace Alignment Index (SENSAI). This is done by respectively maximising and minimizing the subspace similarity index (SSI) of the retained "signal" (SSSI) and removed "noise" (NSSI) with the refCOV (where: SENSAI = SSSI - NSSI).

### Example videos of unsupervised GEDAI denoising


[![Watch a few example videos](https://github.com/user-attachments/assets/6c4145c0-a67c-4575-bd83-e4a40d0dc871)](https://www.youtube.com/watch?v=a4bb2wF_AEk&list=PLFwYJV7JHVFQV1Ki3oxa1BK6forD2TQ1d)


### Prerequisites

* **MATLAB:** The GEDAI code is written for MATLAB.
* **EEGLAB:** An interactive MATLAB toolbox for processing continuous and event-related EEG data. The plugin requires EEGLAB to be installed and running. You can find it at [sccn.ucsd.edu/eeglab/](https://sccn.ucsd.edu/eeglab/).

### Installation

1. Download the full GEDAI plugin as a zip file.
2. Unzip and copy the 'GEDAI-master' folder into the "plugins" subdirectory of EEGlab.
3. Launch MATLAB and then EEGLAB.

### Basic instructions for use

Check out the [Wiki](https://github.com/neurotuning/GEDAI-master/wiki/).

* **Low-cut frequency (`lowcut_frequency`) guidance:**
  * **`0.1` Hz (Default):** Applies a single-stage `0.1` Hz zero-phase Haar wavelet high-pass before broadband cleaning and retains/spatially cleans all spectral wavelet bands (`D_1 .. D_8 + A_8`). Recommended for general EEG and standard baseline-corrected ERP analyses (`P1/N1/P2`, `N400`, `P300`, `LPP`).
  * **`0` Hz (Zero temporal filtering):** Disables wavelet high-pass filtering inside GEDAI (globally zero-centering each channel while spatially cleaning all wavelet bands). Recommended for studying un-baselined pre-stimulus or anticipatory activity (`CNV`, `SPN`, readiness potential), response onset latencies, or when data have already been high-passed with an external causal filter prior to GEDAI.
  * **`0.5` Hz:** Removes all `< 0.5` Hz drift temporally in addition to spatial GEVD cleaning. Optional for resting-state / continuous EEG where no slow event-related potentials or pre-stimulus baselines are analyzed.

## 📜 Citation

If you use GEDAI in your research, please cite the original publication:

*Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering* (2025)  [bioRxiv].  
Ros, T, Férat, V., Huang, Y., Colangelo, C., Kia S.M., Wolfers T., Vulliemoz, S., & Michela, A.  
doi: https://doi.org/10.1101/2025.10.04.680449  

### Watch the walktrough video on Youtube:
[![Watch the Walkthrough video](https://github.com/user-attachments/assets/c1488fa6-236c-4cf8-9483-61728fc60a31)](https://youtu.be/qSM5narynzc)



## License
You may use this software under the terms of the PolyForm Noncommercial License 1.0.0 [LICENSE](LICENSE). This is suitable for personal use, research, or evaluation.

**Commercial License**  
If you wish to use this software in a commercial or proprietary application without being bound by terms of the PolyForm Noncommercial License 1.0.0, you must purchase a commercial license. The core algorithms in this repository are the subject of a pending patent application, and a commercial license includes a grant for patent rights.  

## 📧 Contact

For any questions or enquiries, please contact:
Tomas Ros - tomas.ros@unige.ch

## Acknowledgements
We are gratefully supported by the Center for Biomedical Imaging (CIBM), the Swiss National Science Foundation (SNSF), and Unitec

[![cibm](https://github.com/user-attachments/assets/0e67c6b9-0dae-415f-8321-6b8148862e85)](https://cibm.ch/)
[![snf](https://github.com/user-attachments/assets/9db9e0b5-05dd-488e-9730-9abea2f7e8af)](https://www.snf.ch/en)
[![unitec](https://github.com/user-attachments/assets/5417b534-6d3c-495a-8c24-70f65442221b)](https://www.unige.ch/unitec/)

