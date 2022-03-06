# python tools/visualizations/vis_cam.py \
#     /data/workspace_jack/vehicle_attribute_dataset/source/StandfordCars/checks_2022-02-18/colors/red/standford_012276.jpg \
#     /home/jack/Projects/openmmlab/mmclassification/configs/custom/resnet50_sgd_multi_label_0222.py  \
#     /home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_bce_multi_label_0222/latest.pth \
#     --save-path /home/jack/Projects/openmmlab/mmclassification/work_dirs/debug/11131_11.jpg


python tools/visualizations/vis_cam.py \
    /home/jack/Projects/openmmlab/mmclassification/data/HUST_Vehicle_Color/train/black/11131_11.jpg \
    /home/jack/Projects/openmmlab/mmclassification/configs/custom/resnet50_sgd_bs64_hust.py  \
    /home/jack/Projects/openmmlab/mmclassification/work_dirs/train/resnet50_sgd_bs64_hust/best.pth \
    --save-path /home/jack/Projects/openmmlab/mmclassification/work_dirs/vis/vis_cam/HUST_Vehicle_Color/11131_11.jpg

    # --target-layers 'backbone.layer4.2' \
    # --method GradCAM \
    # --target-category 5 \