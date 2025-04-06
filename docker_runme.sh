docker_image_name=$(docker images | grep onnx | awk '{print $1}')
echo "Running $docker_image_name image"
#----------------------------------------------------------------------------------------------------------------------
docker run -it --gpus all --rm \
    --volume ./:/home/ \
    --workdir /home/ \
    $docker_image_name \
    bash -c "./runtime_cpp/build_and_run_app.sh"