# Jetson-camera-pipeline

To pipe visual window from docker container run this command on a local terminal:
`xhost +local:docker`

In order to run Nvidia example
```
# Move into docker folder
cd docker

# Build the docker container
docker compose up -d

# Enter the docker container
docker exec -it deepstream_dev bash

# Launch the demo
deepstream-app -c models/source1_webcam_peoplenet.txt

```