"""
Project: CL-Detection2024 Challenge Baseline
============================================

This script utilizes the DetectionAlgorithm class to make predictions using a trained model,
and it saves the prediction results as a CSV file.
Email: xiehaoyu2022@email.szu.edu.cn

"""
import init_paths
import argparse
import json
import os.path
from pathlib import Path

import SimpleITK
import cv2
import numpy as np
import pandas as pd
import torch
from evalutils import DetectionAlgorithm
from evalutils.validators import UniquePathIndicesValidator, UniqueImagesValidator

from core import config
from trainers.test_ensemble import LandmarkEnsembleDetector

import re


class Cldetection_alg_2024(DetectionAlgorithm):

    def __init__(self, cfg: config.Config):
        self.input_path = Path("/input/images/processed_images")
        # self.input_path = Path(
        #     "./datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation Set/images/processed_images/")
        model_checkpoint_path = "/opt/algorithm/checkpoints/ensemble.ckpt"
        # model_checkpoint_path = "checkpoints/UNetResnet34/ensemble_test_v3.ckpt"
        # Please do not modify the initialization function of the parent class.
        super().__init__(
            validators=dict(input_image=(UniqueImagesValidator(), UniquePathIndicesValidator())),
            # input_path=Path("/input/images/lateral-dental-x-rays/processed_images"),
            input_path=self.input_path,
            file_filters=dict(input_image=re.compile(r".*\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)),
            output_file=Path("/output/predictions.csv"))

        print("==> Starting...")

        # Use the corresponding GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and weights. The path to the weights file is /opt/algorithm/best_model.pt.
        # In the Docker environment, the current directory is mounted as /opt/algorithm/.
        # Therefore, any file in the current folder can be referenced in the code with the path /opt/algorithm/.

        self.cfg = cfg
        self.model = LandmarkEnsembleDetector.load_from_checkpoint(model_checkpoint_path, cfg=cfg)

        print("==> Using ", self.device)
        print("==> Initializing model")
        print("==> Weights loaded")

    def save(self):
        # All prediction results
        all_images_predict_landmarks_list = self._case_results
        print("==> Predicted")

        # Save the prediction results in the CSV file format required for the challenge.
        columns = ['image file']
        for i in range(53):
            columns.extend(['p{}x'.format(i + 1), 'p{}y'.format(i + 1)])
        df = pd.DataFrame(columns=columns)

        # Iterate through each dictionary and write the data to a CSV file.
        for item in all_images_predict_landmarks_list:
            file_name = item['file name']
            landmarks = item['predict landmarks']
            row_line = [file_name] + [coord for point in landmarks for coord in point]
            df.loc[len(df.index)] = row_line

        df.to_csv(self._output_file, index=False)
        print("==> Saved CSV file")

    def process_case(self, *, idx, case):
        """
        !IMPORTANT: Please do not modify any content of this function. Below are the specific comments.
        """
        # Call the parent class's loading function
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Pass the corresponding input_image in SimpleITK.Image format and return the prediction results.
        predict_one_image_result = self.predict(input_image=input_image,
                                                file_name=os.path.basename(input_image_file_path))

        return predict_one_image_result

    def predict(self, *, input_image, file_name: str):
        """
        :param input_image: The image that needs to be predicted
        :param file_name: The file name of the image
        :return: A dictionary consisting of file names and corresponding prediction values.
        """
        # Convert the SimpleITK.Image format to Numpy.ndarray format for processing.

        image_array = SimpleITK.GetArrayFromImage(input_image)
        
        # Predict
        with torch.no_grad():
            self.model.eval()

            # Image preprocessing operations
            torch_image, image_info_dict = self.preprocess_one_image(image_array, file_name=file_name)
            
            print("==> Predicting", file_name, torch_image.shape)
            # Model prediction
            predictions = self.model.landmark_prediction(torch_image)[0]

            # Post-processing of the results
            predict_landmarks = self.postprocess_model_prediction(predictions, image_info_dict=image_info_dict)
        print("==> Finished", file_name)

        return {'file name': file_name, 'predict landmarks': predict_landmarks}

    def preprocess_one_image(self, image_array: np.ndarray, file_name: str):
        """
        :param image_array: The data that needs to be predicted
        :return: The preprocessed image tensor and image information dictionary
        """

        # Basic information about the image
        image_info_dict = json.load(open(f"{self._input_path}/meta/{file_name.split('.')[0]}_meta.json", "r"))
        # Example of image_info_dict:
        # {"filename": "495", "pixels_per_mm": [1, 1], "shift": [722, 445], "scale_factor": [2.1188455008488964, 2.119318083106837]}

        # Adjust the channel position, add a batch-size dimension, and convert to the torch format.
        # transpose_image_array = np.transpose(image_array, (2, 0, 1))
        if len(image_array.shape) == 3 and image_array.shape[2] in [1, 3]:
            image_array = image_array[:, :, 0]

        torch_image = torch.from_numpy(image_array[np.newaxis, np.newaxis, :, :])

        # Move to a specific device
        torch_image = torch_image.float().to(self.device)

        return torch_image, image_info_dict

    def postprocess_model_prediction(self, predict_heatmap: torch.Tensor, image_info_dict: dict):
        """
        :param predict_heatmap: The predicted heatmap tensor from the model
        :param image_info_dict: Information about the input image
        :return: A list of predicted landmark coordinates
        """

        # Convert to a Numpy matrix for processing: detach gradients, move to CPU, and convert to Numpy.
        predict_heatmap = predict_heatmap.detach().cpu().numpy()
        landmarks = self.model.invert_heatmap_coordinate_to_original_res(predict_heatmap[0],
                                                                         image_info_dict['shift'],
                                                                         np.array(image_info_dict['scale_factor']))
        return landmarks.tolist()


if __name__ == "__main__":
    def parse_args():
        # Create an argument parser
        parser = argparse.ArgumentParser(
            description='Detect landmarks conditional on an image')

        parser.add_argument('--config_path', type=str, help='Path to the configuration file',
                            default="configs/default.yaml", required=False)
        parser.add_argument("--saving_root_dir", type=str, help='Path to where to save project files',
                            default="./", required=False)
        parser.add_argument("--input_images_dir", type=str, help='Path to images',
                            default="/input/images/lateral-dental-x-rays/processed_images", required=False)
        parser.add_argument("--desc", type=str, help="Description of the run", default="", required=False)
        # Parse the command-line arguments
        args = parser.parse_args()
        return args


    args = parse_args()
    # args.config_path = "./configs/docker_configs/rcnn_sub_images.yaml"
    # args.config_path = "./configs/docker_configs/CHH.yaml"
    cfg = config.get_config(args.config_path, args.saving_root_dir)
    algorithm = Cldetection_alg_2024(cfg=cfg)
    algorithm.process()

    # Question: How can we call the process() function if it's not implemented here?
    # Answer: Because Cldetection_alg_2024 inherits from DetectionAlgorithm, it inherits the parent class's functions.
    #         So, when called, it automatically triggers the relevant functions.

    # Question: What operations are performed behind the scenes when calling the process() function?
    # Answer: By referring to the source code, we can see the process() function, which is defined as follows:
    #    def process(self):
    #        self.load()
    #        self.validate()
    #        self.process_cases()
    #        self.save()
    #    We can see that these four functions are executed behind the scenes.
    #    Additionally, within the process_cases() function, the process_case() function is called:
    #    def process_cases(self, file_loader_key: Optional[str] = None):
    #        if file_loader_key is None:
    #            file_loader_key = self._index_key
    #        self._case_results = []
    #        for idx, case in self._cases[file_loader_key].iterrows():
    #            self._case_results.append(self.process_case(idx=idx, case=case))
    #    Therefore, you only need to implement the desired functionality in the process_case() and save() functions.

    # Question: If only process_case() and save() need to be implemented, why is there also a predict() function?
    # Answer: The predict() function is required by the parent class DetectionAlgorithm to predict the results for each case.
    #         Otherwise, it would raise a NotImplementedError.
