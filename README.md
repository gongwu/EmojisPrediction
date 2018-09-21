# SemEval 2018 Task 2: Multilingual Emoji Prediction

## About this code

This is the code by [ECNU](http://aclweb.org/anthology/S18-1068) submitted to SemEval 2018 Task 2. The final model achieves 33.35 Macro
F-score score on the test set and ranks 5th among all the participants. 

## Installation

```
# download the repo
git clone https://github.com/gongwu/SemEval2018-Task2-EmojisPrediction.git
# download the dataset
# run the model
python main.py
```

## Results on Dev

|          Model           | Macro F1 |
| :----------------------: | :------: |
| Traditional NLP Features |  34.63   |
|   Deep Learning Model    |  32.59   |
|    Combination Model     |  35.21   |
|      Ensemble Model      |  35.57   |