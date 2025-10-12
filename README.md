# Flowers-Classification
Flowers-Classification


# 建议新环境
pip install "torch>=2.1" "torchvision>=0.16" "tqdm>=4.65"

# 切换 BiT / ViT 等 timm 模型
pip install "timm>=1.0.7"

# 运行（默认 50 轮、224 分辨率、AMP）
python code/train_flowers102.py --data data/flowers/flowers --timm-model resnetv2_50x1_bit.goog_in21k  --amp --epochs 50 --batch-size 64 --amp
# 不使用timm 直接使用resnet
python code/train_flowers102.py --data data/flowers/flowers   --amp --epochs 50 --batch-size 64 --amp

# 预测脚本
python predict.py --ckpt ./checkpoints_flowers102/best.pth --class-names ./data/flowers/flowers/class_names.json --image-dir ./inference_images --output preds.csv --img-size 288 --batch-size 64 --workers 0 --topk 5