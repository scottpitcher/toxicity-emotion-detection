# multitask_bert.py
"""
This script does the following:
1. Load's in pretrained BERT model
2. creat our models arch (shared BERT Encoder -> toxicity or emotion head)
3. Defines the joint loss fn
4. Implements adjustable loss weighting
5. build the train loop for model
"""