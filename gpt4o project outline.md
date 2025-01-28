### MNIST Training Data Sorting Experiment Plan

#### Objective:
To investigate the impact of training data sorting on neural network learning, interpretability, and performance using the MNIST dataset as a test case.

---
### Software Structure Plan:

#### 1. Directory Structure:
```
mnist_experiment/
|-- data/
|   |-- sorted/
|   |-- random/
|-- src/
|   |-- preprocess.py
|   |-- analysis.py
|   |-- train.py
|   |-- visualize.py
|-- results/
|   |-- activations/
|   |-- confidence_images/
|-- notebooks/
|   |-- exploratory_analysis.ipynb
|-- README.md
|-- requirements.txt
```

#### 2. Components and Their Responsibilities:

##### A. Preprocessing (`preprocess.py`):
- **Module Docstring:**
  """
  Preprocessing module for sorting and shuffling MNIST training data.
  Author: Research Scientist
  """

1. **Sort Training Data**
   - Define sorting orders for labels (e.g., ascending or descending).
   - Separate the MNIST training dataset based on specified label orders.
   - Store sorted datasets in `data/sorted/`.

2. **Generate Random Orders**
   - Shuffle the training dataset randomly to create a baseline.
   - Store random datasets in `data/random/`.

##### B. Analysis Tools (`analysis.py`):
- **Module Docstring:**
  """
  Analysis tools for evaluating neural network performance and interpretability.
  Author: Research Scientist
  """

1. **Heatmap Generation for Output Neurons** (for single-layer NNs):
   - Visualize each output neuron’s activation across the dataset.
   - Save heatmaps in `results/activations/`.

2. **Confidence Analysis** (for complex NNs):
   - Identify test samples with minimum and maximum confidence for each label.
   - Compute the average of all test inputs classified as each label.
   - Save the resulting images in `results/confidence_images/`.

##### C. Training (`train.py`):
- **Module Docstring:**
  """
  Training module for baseline and sorted training experiments on MNIST.
  Author: Research Scientist
  """

1. **Baseline Training:**
   - Train a model using randomly ordered data.

2. **Sorted Training:**
   - Train models using 3+ predefined label orderings.

3. **Save Outputs:**
   - Save training logs, accuracy scores, and model weights for comparison.

##### D. Visualization (`visualize.py`):
- **Module Docstring:**
  """
  Visualization module for generating heatmaps and confidence tables.
  Author: Research Scientist
  """

1. **Plot Activation Heatmaps:**
   - Create a visual representation of each neuron’s activations.

2. **Confidence Image Table:**
   - Generate a 3x10 table showing minimum, maximum confidence samples, and average images for each label.

---

### Experiment Procedure:

#### 1. Data Preparation:
- Load MNIST dataset.
- Use `preprocess.py` to sort training data by:
  - Ascending order of labels.
  - Descending order of labels.
  - A domain-relevant custom order (e.g., 1, 7, 0, 9, 4, 5, 6, 2, 3, 8).

#### 2. Model Design:
- Single-layer NN: No hidden layers, 10 output neurons.
- Complex NN: Multi-layer perceptron with 2 hidden layers.

#### 3. Training:
- Train a baseline model with randomly shuffled data.
- Train 3+ models with sorted data.
- Save metrics and model weights for all experiments.

#### 4. Analysis:
- Visualize neuron activations (single-layer NN).
- Generate confidence images (complex NN).
- Compare metrics (accuracy, loss trends, etc.) across models.

#### 5. Visualization:
- Compile visualizations into a report/notebook.
- Highlight differences between random and sorted data training.

---

### Iteration Plan:
#### Revision Round 1:
- Conduct a detailed review of the software structure, ensuring alignment with the experiment objectives.
- Validate the sorting methods and ensure they match the intended learning trajectory hypothesis.
- Test data preprocessing for correctness and robustness.

#### Revision Round 2:
- Run initial training experiments and analyze preliminary results.
- Adjust visualization tools and add additional metrics if needed (e.g., learning rate trends, confusion matrices).
- Optimize training loops for efficiency using PyTorch and Adam/AdamW optimizers.

#### Revision Round 3:
- Refine all documentation, including module docstrings and inline comments.
- Test reproducibility by running the entire pipeline with fixed random seeds.
- Ensure scalability for larger datasets through code profiling and optimization.

---

### Implementation:

#### Step 1: Finalize the Structure
After revisions, lock in the directory structure and component designs.

#### Step 2: Code Implementation
- Write `preprocess.py` with type hints and detailed docstrings.
- Implement training in `train.py` using PyTorch with Adam/AdamW optimizers.
- Develop analysis tools in `analysis.py` and `visualize.py` with clear type hints and comments.

#### Step 3: Testing
- Test preprocessing with various orders.
- Verify training outputs.
- Validate analysis and visualizations.

#### Step 4: Reporting
- Compile results into a comprehensive report.
- Highlight key findings and insights.

---

### Notes:
- Ensure reproducibility (e.g., random seeds).
- Use modular, reusable code with type hints.
- Keep computational efficiency in mind for scaling to larger datasets.

Would you like to proceed with this structure, or make any adjustments?

