# LipReading
This repo aims to build a Lips Reading model that can read single English alphabet letters and digits 0-9.
The source code is heavily borrowed from [Learn an Effective Lip Reading Model without Pains](https://github.com/Fengdalu/learn-an-effective-lip-reading-model-without-pains). We obtain `85.60%` accuracy on `AVLetters` and `AVDigits` datasets.
We made a notebook on Google Colab for end-to-end pipeline. Check it [here](https://colab.research.google.com/drive/1Stp8yyz1ib1mv1hVPppwpngepPgTphXt)
## Preprocessing
[AVLetters](https://drive.google.com/file/d/1RT1beWiBTyFHt6KlBwjlja9GyOyJ2zed/view) dataset is comprised of 780 `.mat` binary files with file name format like `A1_PERSON-lips.mat`, where the first character is
label (`A-Z`). Each file data has two array `vid` and `siz`. The array `vid` is input video and its shape is `siz` `(60, 80, N)`.

[AVDigits](https://drive.google.com/file/d/1ftS9GHYkOyQ-hQdFvamTblYyIWQCTWud/view) dataset is comprised of 540 `.mp4` files with file name format like `S1_2_01.mp4`, where the `2` character is label (`0-9`).

We handled each dataset in different ways. With `AVLetters` dataset, we only load video frames from files. With `AVDigits` dataset, we extract frames from files and manually crop mouth region by hand.

All frames are padded to squared size `(96, 96)`. Finally, all videos are cut to 40 frames. If video length is less than 40, we just pad it with final frame. For more details, please check `scripts/prepare_dataset.py` script.

The preprocessed dataset could be found [here](https://drive.google.com/file/d/1MFYjuGJqcLp7Ml4_u2zj5dMgzG6qhhxa/view?usp=sharing)

If you want to run preprocessing from scratch. Please download raw data [av.zip](https://drive.google.com/file/d/1MFYjuGJqcLp7Ml4_u2zj5dMgzG6qhhxa/view?usp=sharing). Extract and put it into a folder e.g `datasets`. Run this script to do preprocessing.
```
python scripts/prepare_dataset.py --data-dir datasets/av
```
Processed .pkl files would be generated under `datasets/avletters_digits_npy_gray_pkl_jpeg` folder.

## Training
This works with `pytorch==1.9.0`. Other version might work but haven't tested yet.
Please install all dependencies in `requirements.txt` before running the training script.
For more details, please check `main_visual.py` script.
```
python main_visual.py --save_prefix checkpoints/av-baseline \
                    --dataset av \
                    --data-dir datasets/avletters_digits_npy_gray_pkl_jpeg \
                    --n_class 36 \
                    --batch_size 32 \
                    --se \
                    --label_smooth \
                    --mixup \
```
Model checkpoints should be saved in `checkpoints/` folder. Our best model could be found [here]()

## Evaluation
You can evaluate the trained model easily.
```
python eval.py --dataset av \
                --data-dir datasets/avletters_digits_npy_gray_pkl_jpeg \
                --n_class 36 \
                --weights checkpoints/WEIGHTS.pt \
                --se \
                --label_smooth \
                --mixup
```


## Demo
You can test on a single input video file or a folder contains video files.
```
python demo.py --weights checkpoints/WEIGHTS.pt \
                --n_class 36 \
                --se \
                --label_smooth \
                --mixup \
                --source VIDEO_FILE_OR_FOLDER
```