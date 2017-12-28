import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models, transforms

from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import os
import imutils
from PIL import Image
import PIL
import copy
import logging

logger = logging.getLogger('heatmap')

# TODO Import this from Project X. While importing, replace Infiltration with Infiltrate
l2i = {'Atelectasis': 0,
 'Cardiomegaly': 1,
 'Consolidation': 2,
 'Edema': 3,
 'Effusion': 4,
 'Emphysema': 5,
 'Fibrosis': 6,
 'Hernia': 7,
 'Infiltrate': 8,
 'Mass': 9,
 'No Finding': 10,
 'Nodule': 11,
 'Pleural_Thickening': 12,
 'Pneumonia': 13,
 'Pneumothorax': 14}

class HeatMapGenerator:
    def initialize_with_path(self, model_path):
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            state = torch.load(model_path)
        else:
            state = torch.load(model_path, map_location=lambda storage, loc: storage)

        model = models.resnet50(pretrained=True)

        # changing the final_layer for our images
        num_ftrs = model.fc.in_features
        num_classes = state["fc.weight"].shape[0]
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(state)

        self.initialize_model(model)

    def initialize_model(self, model):
        self.model = model
        self.model.eval()
        self.disease2activationMap_wts = self.model.fc.weight.data
        self.transform = transforms.Compose([
            transforms.ToPILImage(), # TODO Remove this if input image is PIL
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create hook for storing outputs of layers
        self.layer_outputs = None
        def hook(module, input, output):
            self.layer_outputs = output.data
        self.model.layer4[2].conv3.register_forward_hook(hook)  # model.layer4[2].conv3 is the final convolutional layer of resnet

    def get_img_batch(self, image_paths, img_side_length):
        inputs = []
        for img_path in image_paths:
            img = Image.open(img_path)
            img = img.resize((img_side_length, img_side_length), PIL.Image.ANTIALIAS)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)[:, :, ::-1]
            img = self.transform(img)
            inputs.append(img)
        inputs = torch.stack(inputs)
        if self.use_gpu:
            inputs = Variable(inputs, volatile=True).cuda()
        else:
            inputs = Variable(inputs, volatile=True)
        return inputs

    def annotate_img(self, heat_map, img_pixel_wts, min_diff_threshold = 200):
        diff = np.zeros(img_pixel_wts.shape)
        diff[np.where(img_pixel_wts > min_diff_threshold)] = 255
        thresh = cv2.threshold(diff, min_diff_threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if imutils.is_cv2() else contours[1]
        annotated_heatmap = copy.deepcopy(heat_map)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(annotated_heatmap, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return annotated_heatmap

    def generate_heatmap_for_imgPaths(self, image_paths, img_side_length=256, annotate_image=True, heatmap_blend_ratio=0.9, batch_actual_disease=None):
        inputs = self.get_img_batch(image_paths, img_side_length)
        return self.generate_heatmap(inputs, annotate_image, heatmap_blend_ratio, batch_actual_disease)

    def generate_heatmap(self, inputs, annotate_image=True, heatmap_blend_ratio=0.9, batch_actual_disease=None):
        input_img_shape = (inputs.data.shape[2], inputs.data.shape[3])

        # Get weights of activation maps for each image in batch
        outputs = torch.sigmoid(self.model(inputs))
        """
        logger.warning("remove this")
        outputs = outputs.data.numpy()
        outputs_ignore_no_finding = copy.deepcopy(outputs)
        outputs_ignore_no_finding[:, 10] = 0  # Ignore no findings
        batch_pred_disease_index1 = np.argmax(outputs_ignore_no_finding, 1)
        """

        if batch_actual_disease is None:
            outputs = outputs.data.numpy()
            outputs_ignore_no_finding = copy.deepcopy(outputs)
            outputs_ignore_no_finding[:,10] = 0  # Ignore no findings
            batch_pred_disease_index = np.argmax(outputs_ignore_no_finding, 1)
        else:
            batch_pred_disease_index = np.array(batch_actual_disease)

        #print batch_pred_disease_index1 == batch_pred_disease_index


        # Get weighted activation map (8x8)
        batch_map_wts = self.disease2activationMap_wts[batch_pred_disease_index,]  # size: batch_size * 2048
        batch_map_wts = batch_map_wts.unsqueeze(2).unsqueeze(2)  # size: batch_size * 2048 * 1 * 1 (For compatability with torch.mul)

        # Final convolutional layer output
        final_conv_out = self.layer_outputs

        # Get heatmap for each image in batch
        batch_heatmaps = []
        batch_grayscale_heatmaps = []
        for i in range(batch_pred_disease_index.shape[0]):
            # Get weights of each pixel in image
            wted_activation_maps = torch.mul(batch_map_wts[i], final_conv_out[i]);
            wt_activation_map = torch.mean(wted_activation_maps, 0)  # take a mean of the 2048 activation maps
            img_pixel_wts = cv2.resize(wt_activation_map.numpy(), input_img_shape, cv2.INTER_CUBIC)
            img_pixel_wts = (
                cv2.normalize(img_pixel_wts, img_pixel_wts, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F) * 255).astype(np.uint8)
            batch_grayscale_heatmaps.append(img_pixel_wts)
            """cv2.imshow("img_pixel_wts", img_pixel_wts)
            cv2.waitKey(0)"""

            # Represent image pixel weights as heatmap
            hm_weights_colored = cv2.applyColorMap(img_pixel_wts, cv2.COLORMAP_JET)
            img_cv2 = (255 * inputs[i].data.numpy().transpose(1, 2, 0)).astype(np.uint8)

            heat_map = cv2.addWeighted(hm_weights_colored, heatmap_blend_ratio, img_cv2, (1-heatmap_blend_ratio), 0)
            # heat_map = hm_weights_colored

            # Draw bounding box
            if annotate_image:
                heat_map = self.annotate_img(heat_map, img_pixel_wts)

            batch_heatmaps.append(heat_map)

        return batch_heatmaps, batch_grayscale_heatmaps, outputs

# Test
def test_heatmap_generator(model_path, image_names, show_actual_box, img_folder, img_extn, actual_box_diag_coords=None, actual_disease_ids = None):
    img_paths = [os.path.join(img_folder, image_name + img_extn) for image_name in image_names]
    n_images = len(image_names)

    hm_gen = HeatMapGenerator()
    hm_gen.initialize_with_path(model_path)
    batch_heatmaps, batch_grayscale_heatmaps, _ = hm_gen.generate_heatmap_for_imgPaths(img_paths, heatmap_blend_ratio=0.5, batch_actual_disease=actual_disease_ids)
    #exit(0)
    f, axarr = plt.subplots(n_images,2)
    f.set_size_inches(10, 10)
    #f.suptitle("Comparison using model:{0}".format(model_path), fontsize=14)
    f.suptitle("Disease Localization", fontsize=14)
    for i in range(len(batch_heatmaps)):
        heatmap = batch_heatmaps[i]
        input_img = cv2.imread(img_paths[i])
        disease = ""
        if show_actual_box:
            c1, c2, disease = actual_box_diag_coords[i]
            cv2.rectangle(heatmap, c1, c2, (0, 0, 255), 2)

            ix1, iy1, ix2, iy2 = c1[0]*4, c1[1]*4, c2[0]*4, c2[1]*4
            #ix1, iy1, ix2, iy2 = c1[0], c1[1], c2[0], c2[1]
            cv2.rectangle(input_img, (ix1,iy1), (ix2,iy2), (0, 0, 255), 2)

        axarr[i, 0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), cmap='gray')
        axarr[i, 0].set_title("Image {0}-{1}".format(image_names[i], disease))
        axarr[i, 0].axis('off')

        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        axarr[i, 1].imshow(heatmap)
        axarr[i, 1].set_title("Image {0} (Heatmap)".format(image_names[i]))
        axarr[i, 1].axis('off')
        """
        axarr[i, 2].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY), cmap='gray')
        axarr[i, 2].set_title("Image {0} (GrayScale Heatmap)".format(image_names[i]))

        axarr[i, 3].imshow(batch_grayscale_heatmaps[i], cmap='gray')
        axarr[i, 3].set_title("Image {0} (GrayScale Heatmap Original)".format(image_names[i]))
        #plt.imshow(cv2.cvtColor(batch_heatmaps[i], cv2.COLOR_BGR2RGB))
        #plt.pause(10000)
        """

    #plt.show()

def main():
    img_folder = "/Users/vishalrao/Downloads/images/"
    img_extn = ".png"
    # image_names = ["00000344_003", "00000001_000", "00000001_001"]
    # image_names = ["00000032_037","00000072_000","00000147_001","00000149_006","00000150_002","00000181_061","00000193_019","00000211_010","00000211_016"]
    #image_names = ["00000032_037","00000072_000","00000147_001","00000149_006","00000150_002"]
    #image_names = ["00000457_004","00000468_017","00000468_033","00000468_041","00000506_013"]
    #image_names = ["00000211_019","00000211_041","00000377_004","00000398_003"] # Cardiomegaly
    #image_names = ["00000001_000", "00000001_001"]
    #image_names = ["00001555_002","00001688_000","00001836_041","00002290_001","00002578_000"] # Nodule
    #image_names = ["00001153_004","00001170_046","00001248_038","00001320_003"]
    #image_names = ["00000032_037","00000147_001","00000149_006","00000377_004","00000468_017"]
    #image_names = ["00000468_041","00000583_008","00000661_000","00001153_004"]
    # image_names = ["00000661_000","00000732_005","00000506_013","00000740_000"]
    #image_names = ["00000211_019", "00000032_037", "00000072_000", "00000344_003", "00000150_002", "00000830_000"]  # Cardiomegaly, Infiltrate, atelectasis, Effusion, Pneumonia, Mass
    #image_names = ["00000661_000", "00000845_000", "00000147_001","00000377_004"] # Version 1
    #image_names = ["00000661_000", "00000147_001"]  # Version 1
    image_names = ["00000211_016", "00001039_005",# infiltrate
                   "00001248_038",#Pneumothorax
                   #"00001688_000",#Nodule ]
                ]
    n_images = len(image_names)

    ## Get actual box co-ordinates
    show_actual_box = True
    actual_box_diag_coords = None
    localize_for_actual_disease = True
    actual_disease_ids = None
    if show_actual_box:
        box_csv = "/Users/vishalrao/Google Drive/JHU Study Material/Courses/Computer Vision/Project/Reference Material/Dataset/BBox_List_2017.csv"
        properties = pd.read_csv(box_csv, skiprows=1, header=None, low_memory=False, na_filter=False).values
        # list of n_img size containing list of tuples (x1,y1),(x2,y2)
        actual_box_diag_coords = [None] * n_images
        for prop in properties:
            img_name = prop[0].split(".")[0]
            if img_name not in image_names: continue
            index = image_names.index(img_name)
            x1, y1, w, h = int(round(prop[2])) / 4, int(round(prop[3])) / 4, int(round(prop[4])) / 4, int(
                round(prop[5])) / 4
            x2 = x1 + w
            y2 = y1 + h
            disease = prop[1]
            actual_box_diag_coords[index] = [(x1, y1), (x2, y2), disease]
            if localize_for_actual_disease:
                if actual_disease_ids is None: actual_disease_ids = [None] * n_images
                actual_disease_ids[index] = l2i[disease]


    #model_path1 = "/Users/vishalrao/Downloads/projectx_3_0.007314.model"
    #model_path2 = "/Users/vishalrao/Downloads/projectx_10_0.006655.model"
    model_path2 = "/Users/vishalrao/Downloads/projectx_28_0.006506.model"
    #test_heatmap_generator(model_path1, image_names, show_actual_box, img_folder, img_extn, actual_box_diag_coords, actual_disease_ids=actual_disease_ids)
    test_heatmap_generator(model_path2, image_names, show_actual_box, img_folder, img_extn, actual_box_diag_coords, actual_disease_ids=actual_disease_ids)
    print ("Check images")
    plt.show()

if __name__ == "__main__":
    main()
