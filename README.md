

# 运行
python train_timm_efficientv2.py --data-root "/home/featurize/efficientnetv2/datasets" --train-csv "/home/featurize/efficientnetv2/datasets/train_labels.csv" --img-subdir "train" --epochs 80 --final-epochs 12 --img-size 224 --final-img-size 299 --batch-size 64 --amp --freeze-epochs 2 --use-balanced-sampler --use-tta --output "runs/efv2s_flowers"


# 预测脚本
python predict.py --ckpt /home/featurize/efficientnetv2/runs/efv2s_flowers/best.pth --train-dir /home/featurize/efficientnetv2/datasets/train --train-csv /home/featurize/efficientnetv2/datasets/train_labels.csv --n 200 --model tf_efficientnetv2_s --img-size 299 --batch-size 64 --mean 0.485,0.456,0.406 --std  0.229,0.224,0.225 --save-csv results/random200_preds.csv
