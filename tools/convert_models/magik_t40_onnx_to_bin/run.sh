export LD_LIBRARY_PATH=/home/jack/Projects/openmmlab/mmclassification/tools/convert_models/magik_t40_onnx_to_bin:$LD_LIBRARY_PATH
model=./20220308/latest.onnx
./magik-transform-tools \
--framework onnx \
--target_device T40 \
--outputpath ./20220308/latest.mk.h \
--inputpath $model \
--mean 0,0,0 \
--var 255,255,255 \
--img_width 224 \
--img_height 224 \
--img_channel 3 \

# --input_nodes input \
# --output_nodes output