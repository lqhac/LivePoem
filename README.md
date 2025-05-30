# LivePoem
The official repository for the paper *LivePoem: Improving the Learning Experience of Classical Chinese Poetry with AI-Generated Musical Storyboards*, which has been accepted to the **[the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025), Special Track on Human-Centred Artificial Intelligence: Multidisciplinary Contours and Challenges of Next-Generation AI Research and Applications](https://2025.ijcai.org/call-for-papers-human-centred-artificial-intelligence/), 16-22 August, Montreal, Canada.**.  

## Code
In addition to the storyboard generator based on Free-Bloom [(Huang et al., 2022)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/52f050499cf82fa8efb588e263f6f3a7-Abstract-Conference.html), we also release code for the poetry-to-melody generation phase under this repository. The project includes the following steps. For each step, we provide a jupyter notebook file to facilitate step-by-step code execution. 

### STEP 0: Environmental Setup
The environmental configuration is saved to `environment.yml`. Readers may reproduce the environment by `conda create --name <your_env_name> --file environment.yml`.

### STEP 1: Build Dictionary for Lyrics & Melodies
The notebook file `./0_build_dictionary/0_build_dict.ipynb` provides a step-by-step guide to generate the dictionaries for melodies and lyrics, respectively. Running the notebook will create a `music_dict.pkl` file covering the complete vocabulary required for this task.

### STEP 2: Binarise Data Set
Run `./1_binarise_data/1_data_binarisation.ipynb` to binarise the lyrics and melody dataset. The lyrics and melodies should be in .txt and .mid, respectively. We provide some example data under the `./data` directory.

### STEP 3: Train Model
The notebook file `3_train_bart.ipynb` is a step-by-step guide to training the model. Alternatively, running `python 3_train_bart.py` can start the training process in one run.

### STEP 4: Inference
For batch inference of multiple samples, please refer to `4_infer_bart.ipynb`.
For customised single-sample inference, please refer to `4_infer_bart_single.ipynb`.

## User study
We also additionally provide the results from our user study and the codes to perform statistical analyses on these results. Details are included under the folder `./4_subjective_analysis`.  
### Reading Comprehension Test Performance and Self-Assessment Manikin (SAM)
The results of participants' reading comprehension test are stored in `./4_subjective_analysis/test_results.csv`. To perform statistical tests, please refer to `./4_subjective_analysis/statistical_tests.ipynb`.
### Thematic Analysis and Inter-Rater Reliability
We also performed a thematic analysis by coding participants' responses to open-end questions. The results are in `./4_subjective_analysis/response_coding.csv`. For the coded (sub)themes and inter-rater reliability, please refer to `./4_subjective_analysis/thematic_analysis.ipynb`.

## To cite this work
The original manuscript is under this directory named `LivePoem (IJCAI25 HAI Track).pdf`. The final paper will be compiled in the proceedings of IJCAI2025 post the conference taking place. We will update this repository by then accordingly.
```
The official citation will be available post the publication of IJCAI25 proceedings.
```

## Citations
Huang, H., Feng, Y., Shi, C., Xu, L., Yu, J., & Yang, S. (2023). Free-bloom: Zero-shot text-to-video generator with llm director and ldm animator. Advances in Neural Information Processing Systems, 36, 26135-26158.

## Demonstrations of Musical Storyboard
We compile the generated samples within a PPTX file for better illustration purposes. The files can be accessed via this link: https://drive.google.com/drive/folders/1OXtzhETvpkRJRiynQfZchZizyrqQVGox?usp=share_link  
