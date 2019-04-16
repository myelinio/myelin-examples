```bash

docker build -t myelinio/advanced-east-preprocess:v0.1.0 -f Dockerfile.preprocess .

docker run \
-v /Users/ryadhkhsib/Dev/workspaces/nn/myelin-examples/ocr/detection/AdvancedEAST/icpr:/data/icpr \
myelinio/advanced-east-preprocess:v0.1.0

# Preprocess
find ../../floorplan/floorplan_2_east/ -type f -name "11*.jpg" -exec cp {} icpr/image_10000/ \; -print
find ../../floorplan/floorplan_2_east/ -type f -name "11*.txt" -exec cp {} icpr/txt_10000/ \; -print

find rm_floorplans_east -type f -name "*.jpg" -exec cp {} icpr/image_10000/ \; -print
find rm_floorplans_east -type f -name "*.txt" -exec cp {} icpr/txt_10000/ \; -print



docker run \
-v /media/ryadh/DATA4T/Ryadh_data/data/dh-property/icpr:/data/icpr \
myelinio/advanced-east-preprocess:v0.1.0


# Train
docker run \
-v /media/ryadh/DATA4T/Ryadh_data/data/dh-property/models:/models/ \
-v /media/ryadh/DATA4T/Ryadh_data/data/dh-property/icpr:/data/icpr \
-v /home/ryadh/.keras/models/:/root/.keras/models/ \
myelinio/advanced-east:v0.1.0



# Build predict image
 s2i build .  docker.io/myelinio/myelin-deployer-s2i-python:v0.1.2  myelinio/advanced-east-predict:v0.1.0


docker run \
-v /Users/ryadhkhsib/Dev/workspaces/nn/myelin-examples/ocr/detection/AdvancedEAST/saved_model:/models/saved_model \
-v /Users/ryadhkhsib/Dev/workspaces/nn/myelin-examples/ocr/detection/AdvancedEAST/icpr:/data/icpr \
myelinio/advanced-east-predict:v0.1.0

```