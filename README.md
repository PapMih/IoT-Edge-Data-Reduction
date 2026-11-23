This project was developed as part of my Master’s Thesis for the Interdepartmental Postgraduate Program in Electronic Automation at the National and Kapodistrian University of Athens (NKUA). It focuses on designing and implementing a data reduction architecture for multivariate IoT data streams at the network edge, using Python. The system processes continuous ship telemetry data locally and maintains a representative synopsis of the stream in a limited-capacity buffer, reducing the amount of data that needs to be transmitted and stored in the cloud.

The proposed architecture divides the continuous data stream into epochs and uses a Vector CUSUM algorithm to detect statistically significant changes in the system state. When a change is detected, the data retained in the buffer are transmitted to the cloud and a new epoch begins. When the buffer reaches its capacity, three different replacement strategies are investigated to preserve the statistical characteristics and variability of the original data: Max Range, Mahalanobis Range, and Max Distance Expansion.

The system was evaluated using synthetic ship telemetry datasets with temporal correlation, generated using an AR(1) model. A residual-based approach using first differences was applied before the Vector CUSUM algorithm to reduce the effect of temporal dependence. The evaluation examines compression efficiency, preservation of the statistical properties of the original stream, and change-detection performance using metrics including generalized variance, data envelope preservation, compression rate, and PATE-F1, as well as combined evaluation scores for sampling quality, statistical efficiency, and overall performance.

Through this project, I gained practical experience in edge computing, multivariate statistical process control, statistical change detection, object-oriented software design, data processing, and algorithmic implementation for resource-constrained environments.

## Results

The method was evaluated on synthetic multivariate streams (AR(1) model) generated from real sensor data of a container ship, across 120 configurations: decision interval *h* = 6–9 (allowance *k* = 1.0) and *h* = 10–15 (*k* = 0.5), buffer sizes 50 / 100 / 150 / 200, and three replacement strategies.

Data reduction is measured with the compression ratio **C = 1 − M/N**, where *N* is the number of measurements in the original stream and *M* the number of measurements sent to the cloud.

| Configuration | Compression ratio C | Data reduction |
|---|---|---|
| All configurations (range) | 0.548 – 0.909 | 55% – 91% |
| Buffer size 100 (balanced choice when compression is the priority) | 0.813 (mean) | ~81% |
| *h* = 9, *k* = 1.0 (best overall configuration) | 0.887 (mean) | ~89% |
| *h* = 9, *k* = 1.0, buffer 50 (maximum) | 0.909 | ~91% |

**Key findings**

- Increasing *h* makes Vector CUSUM less sensitive, producing fewer epochs and higher compression, with diminishing returns at larger values.
- Compression depends only on the CUSUM parameters and the buffer size, not on the replacement strategy.
- *h* = 9, *k* = 1.0 achieved the highest compression, the best change-detection score (PATE-F1) and the best composite scores.
- **Mahalanobis Range Replacement** preserved the multivariate data envelope best (highest volume and generalized-variance indices) and led all composite scores (SQ, SE, TP) in every configuration, because Mahalanobis distance accounts for correlations and different measurement scales between variables.
- A buffer of 200 gave the best average overall performance; a buffer of 100 is the more balanced choice when compression matters most, with lower variance in performance.

### Limitations

- The evaluation used synthetic data; validation on real, raw ship telemetry is future work.
- The CUSUM parameters (*h*, *k*) are fixed for each stream; adapting them dynamically is a possible extension.

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
