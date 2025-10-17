@echo off
set docker_image_name="onnx"
echo Running %docker_image_name% image
REM ----------------------------------------------------------------------------------------------------------------------
docker run -it --gpus all --rm --volume "%~dp0":/home/ --workdir /home/ %docker_image_name% bash -c "./runtime_cpp/build_and_run_app.sh"