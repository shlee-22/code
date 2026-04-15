# Movement Classification using Joint Angles (SVM)

This repository contains code for classifying lower-limb movement patterns using
hip, knee, and ankle joint angles. The primary goal is to:

1. Train an SVM classifier to distinguish between:
   - **Normal**
   - **Incomplete injury**
   - **Complete injury**
2. Analyze how **implanted conditions**  
   (`Implanted_no sti`, `Implanted_sti`) are positioned relative to the above classes.
3. Visualize movement patterns using:
   - Confusion matrix heatmap
   - Prediction distribution bar charts
   - 3D scatter plots
   - Radar charts (Z-scored features)
   - Parallel coordinates plots

---

## 1. Project Structure

```text
movement_classification/
├── data/
│   └── .xlsx   # Raw movement dataset (Excel)
├── src/
│   ├── __init__.py                         # Marks src as a module
│   ├── config.py                           # Global configuration (paths, constants)
│   ├── constants.py                        # Column names, label name mapping
│   ├── data_loader.py                      # Data loading and train/test split
│   ├── model.py                            # SVM model construction and utilities
│   ├── evaluation.py                       # Evaluation and implanted label analysis
│   ├── visualization.py                    # All plotting utilities
│   └── pipeline.py                         # End-to-end experiment pipeline
├── scripts/
│   └── run_experiment.py                   # Entry point script
├── requirements.txt                        # Python dependencies
└── README.txt