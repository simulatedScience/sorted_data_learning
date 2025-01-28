## Prompt for Gemini 2.0 Flash Thinking Experimental 01-21 (2025)

I'm interested in exploring how sorting training data in machine learning affects the learned policies.
From my knowledge, it has been well established that in supervised learning, you should randomly shuffle the training data.
In reinforcement learning, curriculum learning has lead to significant capability advances though. I argue that in a way, this can be seen as a kind of sorting of the training data for the model.

This research idea was initially inspired by LLM training. The training data for LLMs likely contains texts at all kinds of human skill levels: from kindergarten/ primary school through high school to university and postdoc-levels of intellectual difficulty. Humans learn in this order. However, it seems that LLMs are asked to learn with a random sorting of this data.
There is likely a significant difference in initialization. Human brains are already somewhat capable when they are born, whereas many neural networks start out fully random. Still, I would assume that by learning the data in a more structured way, we might get LLMs to learn more human-like behaviour than with current training schedules.

To investigate this, I want to start with small examples, where it is easy to analyse the learned policies. Start with the MNIST dataset. Create a framework for training dense NNs of given layout (as tuple of layer sizes, default = 0 hidden layers).
Add a preprocessor that sorts the training data in a given way. E.g. all 0s first, then all 1s, etc. or all 9s, then all 8s, then all 7s, etc.
For policy analisis, create a seperate script with two plot types:
- For each agent, generate 10 heatmaps (one for each label), visualising the weights of a NN without hidden layers. Use a seismic colormap in matplotlib.
- For each agent, generate 3 plots showing how confident an agent is in classifications. The plots show the minimum, maximum and average difference between the highest and second highest output of the NN for each label.

Suggested file structure:
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