## GEDAI denoising plugin for EEGLAB

<img width="2048" height="512" alt="GEDAI_logo" src="https://github.com/user-attachments/assets/e7252afd-7619-43de-a36d-6679a14d0023" />

## Generalized Eigenvalue De-Artifacting Instrument (GEDAI) 
**A)** EEG may be considered to be a mixture of electrical activities from a brain “signal” (sub)space, and one containing different types of non-cerebral noise or “artifacts”. This mixture may be “unmixed” by linear decomposition techniques (e.g. PCA or ICA) into separate "components" with individual source locations and respective time-courses. However, although PCA and ICA leverage statistical properties within mixed data to recover underlying sourcesas, they are "blind" source separation methods, functioning without a priori knowledge of the original signals or their mixing process. GEDAI combines theoretical knowledge of the brain’s “signal” subspace with generalized eigenvalue decomposition (GEVD) to automatically separate brain and artifact components. Here, a theoretical model of EEG generation is used as an estimate of the brain’s “noise-free” subspace.


<img width="1094" height="778" alt="GEDAI_summary_fig" src="https://github.com/user-attachments/assets/be039586-40f4-4f71-9d03-846fc4dc6662" />


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

*Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering* (2025)  [bioRxiv]. [DOI/Link to paper]  
Ros, T, Férat, V., Huang, Y., Colangelo, C., Kia S.M., Wolfers T., Vulliemoz, S., & Michela, A. 


## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.  
The open-source license granted here allows for non-commercial, research and evaluation purposes. 

The core algorithms in this repository are the subject of a pending patent application. 

## 📧 Contact

For any questions or for commercial use, please contact:
Tomas Ros - dr.t.ros@gmail.com

## Acknowledgements
We are gratefully supported by the Center for Biomedical Imaging (CIBM) and the Swiss National Science Foundation (SNSF)

![unnamed (1)](https://github.com/user-attachments/assets/a5fe08d8-4066-4cd9-8db5-e8bc41760664)
<img width="238" alt="2022-10-12-134122 670745snsf" src="https://github.com/user-attachments/assets/54d236fd-3745-48a1-b6a2-97ffd9bcd7f1" />

