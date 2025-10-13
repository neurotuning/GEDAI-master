## GEDAI denoising plugin for EEGLAB


![gedai](https://github.com/user-attachments/assets/b42bfd18-3965-495a-85bd-616619d6f641)

## Generalized Eigenvalue De-Artifacting Instrument (GEDAI) 
**A)** EEG may be considered to be a mixture of electrical activities from a brain “signal” (sub)space, and one containing different types of non-cerebral noise or “artifacts”. This mixture may be “unmixed” by linear decomposition techniques (e.g. PCA or ICA) into separate "components" with individual source locations and respective time-courses. However, although PCA and ICA leverage statistical properties within mixed data to recover underlying sourcesas, they are "blind" source separation methods, functioning without a priori knowledge of the original signals or their mixing process. GEDAI combines theoretical knowledge of the brain’s “signal” subspace with generalized eigenvalue decomposition (GEVD) to automatically separate brain and artifact components. Here, a theoretical model of EEG generation is used as an estimate of the brain’s “noise-free” subspace.


![F1 large](https://github.com/user-attachments/assets/2cd836e0-1b22-4b64-8766-026ca1195633)


**B)** Each data covariance matrix (dataCOV) is decomposed into source components with GEVD.  
**C)** The GEVD uses a fixed theoretical reference matrix (refCOV) across all epochs, based on the leadfield matrix of an EEG forward model.  
**D)** To determine the optimal threshold separating brain and artifactual subspaces, output EEG data is evaluated using the Signal & Noise Subspace Alignment Index (SENSAI). This is done by respectively maximising and minimizing the subspace similarities of the retained "signal" and removed "noise" with the refCOV. 

### Example videos of unsupervised GEDAI denoising


[![Watch a few example videos](https://github.com/user-attachments/assets/6c4145c0-a67c-4575-bd83-e4a40d0dc871)](https://www.youtube.com/playlist?list=PLFwYJV7JHVFRmGTpW3y2MqkL_pYTitxpF)


### Prerequisites

* **MATLAB:** The GEDAI code is written for MATLAB.
* **EEGLAB:** An interactive MATLAB toolbox for processing continuous and event-related EEG data. The plugin requires EEGLAB to be installed and running. You can find it at [sccn.ucsd.edu/eeglab/](https://sccn.ucsd.edu/eeglab/).

### Installation

1. Download the full GEDAI plugin as a zip file.
2. Unzip and copy the 'GEDAI-master' folder into the "plugins" subdirectory of EEGlab.
3. Launch MATLAB and then EEGLAB.

## 📜 Citation

If you use GEDAI in your research, please cite the original publication:

*Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering* (2025)  [bioRxiv]. [[DOI/Link to paper](https://www.biorxiv.org/content/10.1101/2025.10.04.680449v1)]  
Ros, T, Férat, V., Huang, Y., Colangelo, C., Kia S.M., Wolfers T., Vulliemoz, S., & Michela, A. 


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

