# In Codice Ratio

## Synthetic dataset generation for sequence prediction models

### Requirements
- numpy
- cv2
- networkx
- matplotlib

### Usage
Set the dataset, corpus and destination paths in the ```generate_textlines.py``` main, then run it to generate synthetic line images and their transcription.

Dataset folder expects structure ```dataset_folder/{character classes}/character_images.png```.

Corpus folder expects structure ```corpus_folder/{text files}.txt```

Files in the destination folder will be of the type ```destination_folder/{i.png, i.txt}``` for each line generated.

json file ```abbr_matchings.json``` maps text to sequences of symbols in the dataset.


## Model training

### Requirements
- tensor2tensor

### Usage
Preprocess generated data (synthetic dataset in the form of i.png, i.txt couples must be in $TMP_DIR/ocr) and put it into $DATA_DIR, using custom problem definition (in ```t2t_usr```):

```
$ t2t-datagen \
    --t2t_usr_dir=t2t_usr \
    --problem=ocr_latin \
    --tmp_dir=$TMP_DIR \
    --data_dir=$DATA_DIR
```

Train the transformer_sketch model on the generated dataset, using custom problem definition (in ```t2t_usr```):

```
$ t2t-trainer \
    --t2t_usr_dir=t2t_usr \
    --problem=ocr_latin \
    --tmp_dir=$TMP_DIR/ocr \
    --data_dir=$DATA_DIR \
    --model=transformer_sketch \
    --hparams_set=transformer_small_sketch \
    --output_dir=$OUTPUT_DIR
```
# Author

This project was developed by [Elena Nieddu](https://github.com/ErisDelaunay) during [Pi School's AI programme](http://picampus-school.com/programme/school-of-ai/) in Fall 2017.
![photo of Elena Nieddu](http://picampus-school.com/wp-content/uploads/2017/11/IMG_2145-2-400x400.jpg)
