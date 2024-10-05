import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import cv2
import matplotlib.colors as mcolors
from torchvision.utils import draw_bounding_boxes

from dataset_utils.dataset_preprocessing_utils import renormalise, get_coordinates_from_heatmap


def plot_img_bounding_box_landmarks(img, bounding_boxes, landmarks, labels=None, convert_to_tensor=True,
                                    show_landmark_indices=False):
    if len(img.shape) == 4:
        b = 0
        img = img[b]
        bounding_boxes = bounding_boxes[b]
        landmarks = landmarks[b]
    img = img.to("cpu")
    bounding_boxes = bounding_boxes.to("cpu")
    landmarks = landmarks.to("cpu")
    if labels is None:
        labels = [""] * bounding_boxes.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(img.shape[2] / 100, img.shape[1] / 100))

    output = draw_bounding_boxes((img * 255).to(torch.uint8), bounding_boxes.float(), labels, width=2)
    ax.imshow(output.permute(1, 2, 0).cpu().numpy())

    boxed_landmarks = torch.round(landmarks).to(torch.int)  # [:, indices]

    ax.scatter(boxed_landmarks[:, 0], boxed_landmarks[:, 1], c='r', s=2)
    if show_landmark_indices:
        for landmark_i, (coordinate_x, coordinate_y) in enumerate(landmarks):
            plt.text(coordinate_x, coordinate_y, str(landmark_i), color='red', fontsize=8)
    ax.set_axis_off()
    ax.margins(x=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # convert mplt image to numpy array
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")
    plt.clf()
    return torch.Tensor(data).permute(2, 0, 1).unsqueeze(0) if convert_to_tensor else data


def plot_heatmaps_and_landmarks_over_img(img, heatmaps, ground_truths, return_as_array=False,
                                         normalisation_method="none"):
    "https://github.com/jfm15/ContourHuggingHeatmaps/blob/main/evaluate.py"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    s = 0
    # print(img.shape, heatmaps.shape, ground_truths.shape)
    # # torch.Size([1, 1, 800, 640]) torch.Size([1, 19, 800, 640]) torch.Size([1, 19, 2])
    # print(img.device, heatmaps.device, ground_truths.device)
    heatmaps = heatmaps * 255
    # Display image
    image = renormalise(img[s, 0], False, method=normalisation_method)  # [1, 1, 800, 640] -> [800, 640]
    ax.imshow(image.cpu().numpy(), cmap="gray")
    # Display heatmaps
    heatmaps_thresh = torch.where(heatmaps > 0.05, heatmaps, 0)
    normalized_heatmaps = heatmaps_thresh / torch.amax(heatmaps_thresh, dim=(2, 3), keepdim=True)

    squashed_heatmaps = torch.amax(normalized_heatmaps, dim=1)
    # squashed_heatmaps = torch.where(squashed_heatmaps > 0.05, squashed_heatmaps, 0)

    squashed_heatmaps_np = squashed_heatmaps[s].cpu().numpy()
    heatmap_min, heatmap_max = np.min(squashed_heatmaps_np), np.max(squashed_heatmaps_np)
    norm = mcolors.Normalize(vmin=heatmap_min, vmax=heatmap_max)
    heatmap_colored = plt.cm.gnuplot2(norm(squashed_heatmaps_np))

    # Display predicted points
    predicted_landmark_positions = get_coordinates_from_heatmap(heatmaps).cpu().numpy()
    # print(predicted_landmark_positions.shape)
    # print(predicted_landmark_positions)

    # Display ground truth points
    ground_truth_landmark_position = ground_truths[s].cpu().numpy()
    ax.scatter(
        ground_truth_landmark_position[:, 1],
        ground_truth_landmark_position[:, 0],
        color="green",
        s=2,
        alpha=0.7,
    )

    ax.scatter(
        predicted_landmark_positions[s, :, 0],
        predicted_landmark_positions[s, :, 1],
        color="red",
        alpha=0.9,
        s=2,
    )

    ax.imshow(heatmap_colored, alpha=0.6)
    ax.set_axis_off()
    if return_as_array:
        ax.margins(x=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # convert mplt image to numpy array
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        plt.clf()
        return data
    return fig


def plot_heatmaps(heatmaps, ground_truths):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    s = 0
    # print(img.shape, heatmaps.shape, ground_truths.shape)
    # # torch.Size([1, 1, 800, 640]) torch.Size([1, 19, 800, 640]) torch.Size([1, 19, 2])
    # print(img.device, heatmaps.device, ground_truths.device)
    heatmaps = renormalise(heatmaps, False)
    # Display heatmaps
    normalized_heatmaps = heatmaps / torch.amax(heatmaps, dim=(2, 3), keepdim=True)

    squashed_heatmaps = torch.amax(normalized_heatmaps, dim=1)

    # Display predicted points
    predicted_landmark_positions = get_coordinates_from_heatmap(normalized_heatmaps).cpu().numpy()
    # print(predicted_landmark_positions.shape)
    # print(predicted_landmark_positions)

    # Display ground truth points
    ground_truth_landmark_position = ground_truths[s].cpu().numpy()
    ax.scatter(
        ground_truth_landmark_position[:, 1],
        ground_truth_landmark_position[:, 0],
        color="green",
        s=2,
        alpha=0.2,
    )

    ax.imshow(squashed_heatmaps[s].cpu().numpy(), cmap="gnuplot2", alpha=0.5)
    ax.set_axis_off()
    return fig


def plot_true_landmarks_over_x_noisy(x_noisy, true_landmarks, plot=False, scale=False):
    """
    Plot ground truth landmarks onto x_t
    """
    landmarks = get_coordinates_from_heatmap(true_landmarks)
    b, c, h, w = x_noisy.shape
    # x_noisy [B, 1, H, W]
    if scale:
        x_noisy = renormalise(x_noisy)
    x_noisy = x_noisy.to(torch.uint8).to("cpu").permute(0, 2, 3, 1).numpy()
    landmarks = landmarks.to("cpu")
    output_img = np.zeros((b, h, w, 3))
    for i in range(b):
        img_color = cv2.cvtColor(x_noisy[i], cv2.COLOR_GRAY2BGR)
        # overlay_original = img_color.copy()
        for landmark in landmarks[i]:
            overlay = img_color.copy()
            cv2.circle(overlay, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)
            img_color = cv2.addWeighted(overlay, 0.6, img_color, 1 - 0.6, 0)
        if plot:
            plt.imshow(img_color.astype(np.uint8))
            plt.show()
        output_img[i] = img_color
    return torch.Tensor(output_img).permute(0, 3, 1, 2)


def plot_landmarks(img: torch.tensor, landmarks: torch.tensor, true_landmark=None, plot=False):
    """
    Plot landmarks and ground truth landmarks onto query image
    """
    import cv2
    # if img.min() < 0:
    # img = renormalise(img).to(torch.uint8)
    img = img.to("cpu")
    landmarks = landmarks.to("cpu")
    if true_landmark is not None:
        true_landmark = true_landmark.to("cpu")
    # img should be B x C x H x W

    if len(img.shape) == 3:
        img = img.reshape(img.shape[0], 1, img.shape[1], img.shape[2])
    img = img.permute(0, 2, 3, 1).clip(0, 255).numpy().astype(np.uint8)
    output_img = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
    for i in range(img.shape[0]):
        img_color = cv2.cvtColor(img[i], cv2.COLOR_GRAY2BGR)
        if true_landmark is not None:
            for landmark in true_landmark[i]:
                if landmark[0] < 0 or landmark[1] < 0 or landmark[1] >= img.shape[1] or landmark[0] >= img.shape[2] or \
                        torch.isnan(landmark[0]) or torch.isnan(landmark[1]):
                    continue
                cv2.circle(img_color, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

        for landmark in landmarks[i]:
            if landmark[0] < 0 or landmark[1] < 0 or landmark[1] >= img.shape[1] or landmark[0] >= img.shape[2] or \
                    torch.isnan(landmark[0]) or torch.isnan(landmark[1]):
                continue
            cv2.circle(img_color, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)

        if plot:
            plt.imshow(img_color.astype(np.uint8))
            plt.show()

        output_img[i] = img_color
    return torch.tensor(output_img).permute(0, 3, 1, 2)


def plot_landmarks_from_img(img: torch.tensor, landmarks: torch.tensor, true_landmark=None, plot=False):
    """
    Plot landmarks and ground truth landmarks onto query image given image coordinates
    """
    landmarks = get_coordinates_from_heatmap(landmarks)
    if true_landmark is not None:
        true_landmark = get_coordinates_from_heatmap(true_landmark)
    return plot_landmarks(img, landmarks, plot=plot, true_landmark=true_landmark)


def visualise_full_size_submissions(submission_file, folder, gt_label_file):
    from skimage import io as sk_io
    from skimage import draw as sk_draw

    submission_df = pd.read_csv(submission_file)
    gt_labels = pd.read_csv(gt_label_file)
    training_files = os.listdir(folder)
    training_files = [folder + "/" + file for file in training_files if file.endswith(".bmp")]
    validation_files = os.listdir(
        "datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation Set/images")
    validation_files = [
        "datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Validation Set/images/" + file for file
        in validation_files if file.endswith(".bmp")]
    files = training_files + validation_files

    new_folder_name = submission_file + " Comparison Images"
    if not os.path.exists(new_folder_name):
        os.mkdir(new_folder_name)
    euclidean_dist = np.array([])
    sdr = 0
    total_landmarks = 0
    for img_file_full in sorted(files):
        img_file = img_file_full.split("/")[-1]
        if img_file.endswith(".bmp"):

            if len(submission_df.loc[submission_df["image file"] == f"{img_file}"]) == 0:
                continue
            image = sk_io.imread(img_file_full, cv2.IMREAD_GRAYSCALE)
            landmarks_submission = submission_df.loc[submission_df["image file"] == f"{img_file}"].iloc[0,
                                   1:].values.astype('float').reshape(-1, 2)
            landmarks_gt = gt_labels.loc[gt_labels["image file"] == f"{img_file}"].iloc[0, 2:].values.astype(
                'float').reshape(-1, 2)
            total_landmarks += landmarks_gt.shape[0]
            euclidean_distance = np.linalg.norm(landmarks_submission - landmarks_gt, axis=1)
            scaled_euclidean_distance = np.linalg.norm(
                (landmarks_submission - landmarks_gt) * gt_labels.loc[gt_labels["image file"] == f"{img_file}"].iloc[
                    0, 1], axis=1)
            print(
                f"{img_file} - Euclidean Distance: {euclidean_distance.mean()} Scaled Euclidean Distance: {scaled_euclidean_distance.mean()}")
            euclidean_dist = np.append(euclidean_dist, scaled_euclidean_distance)
            sdr += np.sum(scaled_euclidean_distance < 2)
            # fig, ax = plt.subplots(1, 1, figsize=(image.shape[1] / 150, image.shape[0] / 150))
            #
            # ax.imshow(image, cmap="gray")
            # ax.scatter(landmarks_submission[:, 0], landmarks_submission[:, 1], c="red", s=10)
            # # ax.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], c="red", s=10, alpha=0.5)
            #
            # ax.set_axis_off()
            # ax.margins(x=0)
            # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # # ax.set_title(img_file)
            # # plt.imsave("submissions/Comparison Images/" + img_file, fig)
            # plt.savefig(new_folder_name + "/" + img_file.split(".")[0] + ".png")
            # # plt.show()
            # plt.close()
            # plt.clf()

            image_shape = np.shape(image)[:2]
            for i in range(np.shape(landmarks_gt)[0]):
                landmark, predict_landmark = landmarks_gt[i, :], landmarks_submission[i, :]
                # ground truth landmark
                radius = 7
                rr, cc = sk_draw.disk(center=(int(landmark[1]), int(landmark[0])), radius=radius, shape=image_shape)
                image[rr, cc, :] = [0, 255, 0]
                # add text for landmark index
                cv2.putText(image, str(i), (int(landmark[0]), int(landmark[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                # model prediction landmark
                rr, cc = sk_draw.disk(center=(int(predict_landmark[1]), int(predict_landmark[0])), radius=radius,
                                      shape=image_shape)
                image[rr, cc, :] = [255, 0, 0]
                # the line between gt landmark and prediction landmark
                line_width = 5
                rr, cc, value = sk_draw.line_aa(int(landmark[1]), int(landmark[0]), int(predict_landmark[1]),
                                                int(predict_landmark[0]))
                for offset in range(line_width):
                    offset_rr, offset_cc = np.clip(rr + offset, 0, image_shape[0] - 1), np.clip(cc + offset, 0,
                                                                                                image_shape[1] - 1)
                    image[offset_rr, offset_cc, :] = [255, 255, 0]

            # filename = os.path.basename(file_path)
            if "val" in img_file_full.lower():
                sk_io.imsave(new_folder_name + "/val_" + img_file.split(".")[0] + ".png", image)
            else:
                sk_io.imsave(
                    new_folder_name + "/" + img_file.split(".")[
                        0] + f"-mre={scaled_euclidean_distance.mean():.2f}" + ".png",
                    image)

    print(np.mean(euclidean_dist))
    print(sdr / total_landmarks)


def visualise_wandb_sweep_data():
    import wandb

    # Load sweep data
    api = wandb.Api()
    # sweep_id = "gjdbztxz"
    # config_names = ["ENCODER_CHANNELS", "USE_LAP_PYRAMID", "BLOCKS_PER_LEVEL"]
    sweep_id = "xojz0eyy"
    config_names = ["NAME", "GRAYSCALE_TO_RGB"]
    project = "DiffLand"
    sweep = api.sweep(project + "/" + sweep_id)
    runs = sweep.runs

    for config_name in config_names:

        for summary_metric in ["val/l2_scaled", "val/sdr_l2_scaled_2.0"]:
            # Extract data

            y_values = []
            x_values = []
            for run in runs:
                try:
                    y_values.append(run.summary[summary_metric])
                    x_values.append(run.config["DENOISE_MODEL"][config_name])
                except KeyError:
                    continue
            # Convert to integers (optional)
            mapping = dict()
            counter = 0

            for run in runs:
                value = run.config["DENOISE_MODEL"][config_name]
                # print(value, type(value), isinstance(value, (int, float)))
                if type(value) == bool:
                    if value not in mapping:
                        mapping[value] = counter
                        counter += 1
                    continue
                elif type(value) == str:
                    if value not in mapping:
                        mapping[value] = counter
                        counter += 1
                    continue
                if isinstance(value, (int, float)):
                    continue  # Skip numeric values
                if isinstance(value, list):
                    value = tuple(value)
                if value not in mapping:
                    mapping[value] = counter
                    counter += 1
            if config_name == "USE_LAP_PYRAMID":
                x_values = [mapping.get(value, value) for value in x_values]
            elif config_name == "GRAYSCALE_TO_RGB":
                x_values = [mapping.get(value, value) for value in x_values]
            elif config_name == "NAME":
                x_values = [mapping.get(value, value) for value in x_values]
            else:

                x_values = [mapping.get(tuple(value), counter) for value in x_values]

            # Create scatter plot
            plt.scatter(x_values, y_values)
            if summary_metric == "val/l2_scaled":
                plt.ylim(1, 1.5)
                plt.yscale("log")
            elif summary_metric == "val/sdr_l2_scaled_2.0":
                plt.ylim(80.5, 84)
            plt.xlabel(f"{config_name}")
            plt.ylabel(f"{summary_metric}")
            plt.title("Scatter plot of " + summary_metric + " vs " + config_name)
            plt.show()
            print(mapping)


if __name__ == "__main__":
    # plt.rcParams["figure.figsize"] = (10, 10)
    # plt.rcParams["image.cmap"] = "gray"
    plt.rcParams["figure.dpi"] = 100
    # file = "predictions_chh_new_new_new_hist_norm_on_rcnn.csv"
    # file = "predictions_adv_2.0.4.csv"
    # file = "predictions_multi_image_2.0.6_0szrgf0u.csv"
    # 0.6732469465289231
    # 0.9283018867924528
    # file = "predictions_multi_image_2.0.4_euhi1rv7.csv"

    # file = "predictions_chh_cross_val_best_fold.csv"
    for file in [
        "predictions_ensemble_post_challenge_chh.csv",
        "predictions_ensemble_post_challenge_chh_convnext.csv",
        "predictions_ensemble_post_challenge_nano.csv",
        "predictions_ensemble_post_challenge_tiny.csv"
    ]:
        # file = "predictions_convnext_tiny_post_challenge_validation.csv"
        visualise_full_size_submissions(f"submissions/{file}",
                                        "datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images",
                                        "datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/labels_with_val_estimates.csv")
    # visualise_wandb_sweep_data()
