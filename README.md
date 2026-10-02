This project was developed as part of my Master’s Thesis for the Interdepartmental Postgraduate Program in Electronic Automation at the National and Kapodistrian University of Athens (NKUA). It focuses on designing and implementing a data reduction architecture for multivariate IoT data streams at the network edge, using Python. The system processes continuous ship telemetry data locally and maintains a representative synopsis of the stream in a limited-capacity buffer, reducing the amount of data that needs to be transmitted and stored in the cloud.

The proposed architecture divides the continuous data stream into epochs and uses a Vector CUSUM algorithm to detect statistically significant changes in the system state. When a change is detected, the data retained in the buffer are transmitted to the cloud and a new epoch begins. When the buffer reaches its capacity, three different replacement strategies are investigated to preserve the statistical characteristics and variability of the original data: Max Range, Mahalanobis Range, and Max Distance Expansion.

The system was evaluated using synthetic ship telemetry datasets with temporal correlation, generated using an AR(1) model. A residual-based approach using first differences was applied before the Vector CUSUM algorithm to reduce the effect of temporal dependence. The evaluation examines compression efficiency, preservation of the statistical properties of the original stream, and change-detection performance using metrics including generalized variance, data envelope preservation, compression rate, and PATE-F1, as well as combined evaluation scores for sampling quality, statistical efficiency, and overall performance.

Through this project, I gained practical experience in edge computing, multivariate statistical process control, statistical change detection, object-oriented software design, data processing, and algorithmic implementation for resource-constrained environments.

## Repository Layout

* 01_mcusum_core/: Core implementation of the data buffer, Vector CUSUM, and replacement strategies.
* 02_dataset_creation/: Scripts for generating the synthetic ship telemetry datasets.
* 03_simulation/: Monte Carlo simulation scripts used for Vector CUSUM parameter calibration and ARL₀ analysis.
* 04_evaluation/: Scripts for evaluating the performance of the implemented algorithms.
* 05_analysis/: Scripts for generating summary statistics from the evaluation results.
* 06_tests/: Unit tests for the core components and mathematical operations.

## Running the Project

1. Install the dependencies:

   pip install -r requirements.txt

2. Generate the synthetic datasets by running the scripts in `02_dataset_creation/`.
   *Note: Generated data files are intentionally excluded from the repository to keep it lightweight.*

3. Run the evaluation scripts in 04_evaluation/.

4. If required, run the scripts in 05_analysis/ to generate summary statistics from the evaluation results.
