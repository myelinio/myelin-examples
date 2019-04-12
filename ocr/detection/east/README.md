train docker run:
```bash
docker run \
-v /home/ryadh/Dev/workspaces/nn/text_localization/EAST/model/east_icdar2015_resnet_v1_50_rbox:/tmp/east_icdar2015_resnet_v1_50_rbox/ \
-v /tmp/east:/model/ \
-v /media/ryadh/DATA4T/Ryadh_data/data/dh-property/rm_floorplans_east:/data/EAST/train \
-e NUM_READERS=2 \
-e GPU_LIST=1 \
myelinio/east-train:v0.1.0
```