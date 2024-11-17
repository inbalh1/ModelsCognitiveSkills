# ModelsCognitiveSkills
To create the conda env:
conda create --prefix=models_cognitive_skills --clone /home/morg/students/danaramati/miniconda3/envs/

### The directories:
- results - results from running the models (including csv file with predictions)
- analysis - analysis of the predictions, including graphs, visualization and statistical tests. The directory contains a folder for each model, as well as results for the united data across models.
- analysis\ united graphs - graphs and visualizations for all the models together.

### python files:
eval_on_dataset.py - the old script:
calculates correct prediction by string comparison also plotting the results for a single model.

eval_model_on_sentences.pt - the new script for handout.
calculates correct prediction based on the first different tokens probabilities, and do not plot graphs for a single model.