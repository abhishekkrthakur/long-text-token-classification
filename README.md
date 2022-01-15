# Long text token classification using LongFormer

The data comes from: https://www.kaggle.com/c/feedback-prize-2021/

To train the model for 5 folds, you can run:

    python train.py --fold 0 --model allenai/longformer-large-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4
    python train.py --fold 1 --model allenai/longformer-large-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4
    python train.py --fold 2 --model allenai/longformer-large-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4
    python train.py --fold 3 --model allenai/longformer-large-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4
    python train.py --fold 4 --model allenai/longformer-large-4096 --lr 1e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4

Note that you need have `kfold` column in training data.
