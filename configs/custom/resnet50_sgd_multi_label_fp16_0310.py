_base_ = ['./resnet50_sgd_multi_label_0310.py']

# fp16 settinjgs
fp16 = dict(loss_scale='dynamic')