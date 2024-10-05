# Generate RCNN images
python core/rcnn_extraction.py --config_path /opt/algorithm/configs/docker_configs/rcnn_full_images.yaml --input_images_dir /input/images/lateral-dental-x-rays/
# detect new images
python core/process.py --config_path /opt/algorithm/configs/main_config.yaml