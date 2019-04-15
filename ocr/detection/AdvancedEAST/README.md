```bash

docker run \
-v /Users/ryadhkhsib/Dev/workspaces/nn/myelin-examples/ocr/detection/AdvancedEAST/icpr:/data/icpr \
myelinio/advanced-east-preprocess:v0.1.0

# Preprocess
docker run \
-v /media/ryadh/DATA4T/Ryadh_data/data/dh-property/icpr:/data/icpr \
myelinio/advanced-east-preprocess:v0.1.0


# Train
docker run \
-v /media/ryadh/DATA4T/Ryadh_data/data/dh-property/models:/models/ \
myelinio/advanced-east:v0.1.0







```