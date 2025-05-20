# LivePoem
Official repository for the paper *LivePoem: Improving the Learning Experience of Classical Chinese Poetry with AI-Generated Musical Storyboards*, which has been accepted to the **Human-Centred AI Track, the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025)**.  

## Demonstrations of Musical Storyboard
We have compiled the generated samples within a PPTX file for better illustration purposes. The files can be accessed via this link: https://drive.google.com/drive/folders/1OXtzhETvpkRJRiynQfZchZizyrqQVGox?usp=share_link  

## Code
In addition, we also release code for the poetry-to-melody generation phase under this repository. The project includes the following steps. For each step, we provide a jupyter notebook file to facilitate step-by-step code execution. 

### Build Dictionary for Lyrics & Melodies
Please run the jupyter notebook file "./0_build_dictionary/0_build_dict_octuple.ipynb" will generate the used dictionary as music_dict.pkl under binary folder. 

### Binarise Data
Run ./1_binarise_data/1_octuplemidi.ipynb to binarise the lyrics and melody dataset

### Train Model
```python
python ./2_train_model/3_train_bart_octuple.py
```

### Inference
For batch inference, please use "4_infer_bart_octuple.ipynb"
For custom single-sample inference, please use "4_infer_bart_octuple_single.ipynb"
