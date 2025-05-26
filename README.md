# LivePoem
The official repository for the paper *LivePoem: Improving the Learning Experience of Classical Chinese Poetry with AI-Generated Musical Storyboards*, which has been accepted to the **the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025), Special Track on Human-Centred Artificial Intelligence: Multidisciplinary Contours and Challenges of Next-Generation AI Research and Applications, 16-22 August, Montreal, Canada.**.  

## Demonstrations of Musical Storyboard
We have compiled the generated samples within a PPTX file for better illustration purposes. The files can be accessed via this link: https://drive.google.com/drive/folders/1OXtzhETvpkRJRiynQfZchZizyrqQVGox?usp=share_link  

## Code
In addition to the storyboard generator based on Free-Bloom (Huang et al., 2022), we also release code for the poetry-to-melody generation phase under this repository. The project includes the following steps. For each step, we provide a jupyter notebook file to facilitate step-by-step code execution. 

### Build Dictionary for Lyrics & Melodies
The notebook file `./0_build_dictionary/0_build_dict_octuple.ipynb` provides a step-by-step guide to generate the dictionaries for melodies and lyrics, respectively. Running the notebook will create a `music_dict.pkl` file covering the complete vocabulary required for this task.

### Binarise Data
Run `./1_binarise_data/1_octuplemidi.ipynb` to binarise the lyrics and melody dataset.

### Train Model
The notebook file `3_train_bart.ipynb` is a step-by-step guide to training the model. Alternatively, running `python 3_train_bart.py` can start the training process in one go.

### Inference
For batch inference of multiple samples, please run `4_infer_bart.ipynb`.
For customised single-sample inference, please run `4_infer_bart_single.ipynb`.

## To cite this work
```
COMING SOON
```

## Citations
Huang, H., Feng, Y., Shi, C., Xu, L., Yu, J., & Yang, S. (2023). Free-bloom: Zero-shot text-to-video generator with llm director and ldm animator. Advances in Neural Information Processing Systems, 36, 26135-26158.
