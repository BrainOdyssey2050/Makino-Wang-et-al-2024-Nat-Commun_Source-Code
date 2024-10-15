# **Multi-Regional Control of Amygdalar Dynamics Reliably Reflects Fear Memory Age**

## Code and Demonstrations

- **Analysis of power, phase-amplitude coupling and instantaneous amplitude correlation**:
  - Check the `Correlation_and_Modulation_Analysis` folder for the following demos with expected outputs:
    - `PSD_LFP.m`: Calculates the power spectral density of LFP.
    - `CFC_Phase_Amp.m`: Calculates cross-frequency phase-amplitude coupling of LFP.
    - `Corr_EnvCorr_Env.m`: Calculates the correlation of instantaneous amplitude of LFP across 3 regions.

- **LightGBM Analysis**:
  - For non-freezing or freezing data analysis, visit `MemoryAge_WinLen_5_Step_0.5/code` or `MemoryAge_WinLen_5_Step_0.5_FREEZING_ONLY/code`. These folders contain files for modeling and analysis notebooks with expected outputs for figure generation (notebooks contain 'Figure' in their names). Follow the STEPS provided.

- **Transformer Analysis**:
  - For non-freezing or freezing data analysis, visit `MemoryAge_Transformer_xLARGE/code` or `MemoryAge_Transformer_xLARGE_using_FREEZING_ONLY/code`. Key files include:
    - `MemoryAge ViT CrossSubject NonFreezing.ipynb`: Main file for Transformer training.
    - `Using_Attention_to_analyze_Transformer_decision - Comparing Recent Remote in Accurate (or Inaccurate) - BroaderWelch - without Corr - Figure 5_5.ipynb`: For attention-guided analysis.
    - `Perturbation_results_analysis_Unnormalized - Figure6_2.ipynb`: For perturbation analysis.
	- Please also check the analysis in notebooks with the expected outputs (notebooks contain 'Figure' in their names).
  - Note: It is recommended not to rerun the notebooks as the outputs are already displayed.

## Reproduction Requirements
- **System Requirements**:
  - OS: Windows 10 Professional (Version 22H2)
  - Code Editor: VSCode (Version 1.88.1)
  - Package Manager: Conda (Version 4.10.3)
- **Hardware Requirements**:
  - CPU: Intel(R) Core(TM) i5-9400F @ 2.90GHz
  - GPU: Nvidia GeForce 2080Ti
- **Setup**:
  - Conda environment installation: Run `conda env create -f LFP_TRANS_LOCAL.yml` (takes approx. 10 mins).
- **Expected Runtime**:
  - LightGBM: Approximately 30 minutes.
  - Transformer: Approximately 12 hours (depending on the hardware).
- **Acknowledgments**:
  - We utilize the backbone IO strategy from [TorchEEG](https://github.com/torcheeg/torcheeg).

