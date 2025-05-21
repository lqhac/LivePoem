# LivePoem
Official repository for the paper *LivePoem: Improving the Learning Experience of Classical Chinese Poetry with AI-Generated Musical Storyboards*, which has been accepted to the **Human-Centred AI Track, the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)**.  

## Demonstrations of Musical Storyboard
We have compiled the generated samples within a PPTX file for better illustration purposes. The files can be accessed via this link: https://drive.google.com/drive/folders/1OXtzhETvpkRJRiynQfZchZizyrqQVGox?usp=share_link  

## Code
In addition, we also release code for the poetry-to-melody generation phase under this repository. The project includes the following steps. For each step, we provide a jupyter notebook file to facilitate step-by-step code execution. 

### Build Dictionary for Lyrics & Melodies
The notebook file `./0_build_dictionary/0_build_dict_octuple.ipynb` provides a step-by-step guide to generate the dictionaries for melodies and lyrics, respectively. Running the notebook will create a `music_dict.pkl` file covering the complete vocabulary required for this task.

### Binarise Data
Run `./1_binarise_data/1_octuplemidi.ipynb` to binarise the lyrics and melody dataset.

### Train Model
The notebook file `3_train_bart.ipynb` is a step-by-step guide to training the model. Alternatively, running `python 3_train_bart.py` can start the training process in one go.

### Inference
For batch inference, please use `4_infer_bart.ipynb`
For custom single-sample inference, please use `4_infer_bart_single.ipynb`
