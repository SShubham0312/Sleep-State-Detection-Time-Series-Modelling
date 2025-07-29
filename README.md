# Sleep State Detection using Time-Series modelling

## 1. Data Preprocessing
- **Data Loading**:
  - The dataset is downloaded from Kaggle and extracted if not already present.
  - Training and test series are loaded using `pyarrow` for efficient data handling.

- **Handling Missing Data**:
  - Missing values in critical columns (`anglez`, `enmo`, `date_time`) are dropped during preprocessing.

- **Tolerance Labelling**:
  - Data is truncated around the events 'onset' and wakeup to handle class imabalances.
  - Events (`onset` and `wakeup`) are labeled using two approaches:
    - **Hard Labels**: Events are spread over a window (`2 * decay + 1`).
    - **Soft Labels**: Linear decay is applied to event probabilities over a specified window, ensuring smooth transitions.

- **Event Extraction**:
  - Data is sliced around events with padding to focus on relevant time windows.

---

## 2. Feature Engineering
- **Time-Based Features**:
  - Extracted components like `hour`, `minute`, and `month` from the `date_time` column.
  - Cyclic encodings (`sin` and `cos`) for time-based features to capture periodicity.

- **Lag/Lead Features**:
  - Differences between consecutive values (`lag_diff` and `lead_diff`) for `anglez` and `enmo`.

- **Rolling Statistics**:
  - Calculated rolling mean, max, and standard deviation over a 60-step window for `anglez` and `enmo`.

- **Normalization**:
  - Features are normalized to a range of `[0, 1]` for consistent scaling.

---

## 3. Model Architecture
- **BiLSTM Classifier**:
  - A bidirectional LSTM (BiLSTM) is used to capture temporal dependencies in the data.
  - The architecture includes:
    - **LSTM Layer**: Bidirectional LSTM with hidden states concatenated.
    - **Fully Connected Layer**: Maps the concatenated hidden states to class probabilities.

- **Training Setup**:
  - Input sequences are created with a fixed length (`seq_len`).
  - Data is split into training (80%) and validation (20%) sets.
  - Loss function: Cross-Entropy Loss.
  - Optimizer: AdamW with a learning rate of `1e-3`.

---

## 4. Evaluation
- **Metrics**:
  - Accuracy, Precision, Recall, and F1-Score (macro and class-wise) are computed for validation.
  - Class-wise metrics are tracked across epochs.

- **Visualization**:
  - Training and validation loss curves are plotted to monitor convergence.
  - Class-wise precision and validation metrics (accuracy, precision, recall, F1) are visualized over epochs.
  - Predictions vs. actual labels are plotted for qualitative evaluation.

---

This pipeline effectively preprocesses the data, engineers meaningful features, trains a BiLSTM model, and evaluates its performance using robust metrics and visualizations.
