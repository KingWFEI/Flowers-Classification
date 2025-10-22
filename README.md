# Flowers-Classification
Flowers-Classification


# 建议新环境
pip install "torch>=2.1" "torchvision>=0.16" "tqdm>=4.65"

# 切换 BiT / ViT 等 timm 模型
pip install "timm>=1.0.7"



# 预测脚本
python predict.py --ckpt ./checkpoints_flowers102/best.pth --class-names ./data/flowers/flowers/class_names.json --image-dir ./inference_images --output preds.csv --img-size 288 --batch-size 64 --workers 0 --topk 5

# mean 和 std
mean: [0.45134347677230835, 0.4673071503639221, 0.3222246766090393] std: [0.24617701768875122, 0.22343231737613678, 0.2512664794921875]

# 训练脚本
python src/train.py  --amp --epochs 50 --batch-size 64

python train.py  --amp --epochs 50 --batch-size 64 

python train.py --amp --epochs 50 --batch-size 64 --use-fadc

# 预测主流程 带mean和std
python predict_folder.py \
  --ckpt checkpoints_flowers102/best.pth \
  --class-names data/flowers_train_images/class_names.json \
  --image-dir path/to/test_images \
  --output submission.csv \
  --img-size 288 \
  --mean 0.45134348,0.46730715,0.32222468 \
  --std  0.24617702,0.22343232,0.25126648

python train_test.py --data-root data --train-csv data/train_labels.csv --img-subdir train_images --epochs 80 --final-epochs 12 --img-size 224 --final-img-size 299 --batch-size 64 --amp --use-balanced-sampler --dataset-mean 0.45134347677230835,0.4673071503639221,0.3222246766090393 --dataset-std  0.24617701768875122,0.22343231737613678,0.2512664794921875 --use-snapmix --snapmix-alpha 5.0 --snapmix-prob 0.5 --freeze-epochs 2 --use-tta --output runs/exp_sota
