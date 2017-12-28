import cv2
from matplotlib import pyplot as plt, figure
from skimage.measure import compare_ssim
import argparse
import imutils
import numpy as np
import os
import logging
import copy
from HeatMapGenerator import HeatMapGenerator
#from ProjectX import l2i

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Compare X-Ray Images")
parser.add_argument("--image_dir", required=True,help="X-ray image directory")
parser.add_argument("--img1", required=True, help="Image 1 path")
parser.add_argument("--img2", required=True, help="Image 2 path")
parser.add_argument("--model_path", required=True, help="Path of the model to be used for heat map generation")
parser.add_argument("--diff_threshold", required=False, type=int, default=0.6, help="Pixels with SSIM less than this value would be highlighted (Default = 0.6)")
#parser.add_argument("--img_resize_ratio", required=False, type=float, default=1, help="Image resize ratio (default = 1)")
parser.add_argument("--opening_kernel_size", required=False, type=int, default=10, help="Opening kernel size for post-processing of threshold image (default = 10)")
parser.add_argument("--ignore_border_x_pc", required=False, type=float, default=10, help="Width % of the ignore-region near the vertical borders (default = 9%)")
parser.add_argument("--ignore_border_y_pc", required=False, type=float, default=10, help="Height % of the ignore-region near the horizontal borders (default = 9%)")
parser.add_argument("--heatmap_threshold", required=False, type=int, default=220, help="Heatmap Threshold (default 200)")
parser.add_argument("--output_img_side_length", required=False, type=int, default=256, help="Output image side length (default 256)")
parser.add_argument("--max_heat_grad_diff_for_sim", required=False, type=int, default=50, help="Max heat grad diff for similarity (default 50)")
parser.add_argument("--heatmap_blend_ratio", required=False, type=float, default=0.5, help="Heatmap blend ratio (default 0.5)")
parser.add_argument("--top_n_diseases", required=False, type=int, default=5, help="top n diseases (default 5)")

# TODO Use for output_img_side_length for heatmap and all other images
# TODO Check how the heatmap should be displayed when 'No finding' is the most probable one. Show image for most likely disease instead?
# TODO Import this from Project X
l2i = {'Atelectasis': 0,
 'Cardiomegaly': 1,
 'Consolidation': 2,
 'Edema': 3,
 'Effusion': 4,
 'Emphysema': 5,
 'Fibrosis': 6,
 'Hernia': 7,
 'Infiltration': 8,
 'Mass': 9,
 'No Finding': 10,
 'Nodule': 11,
 'Pleural_Thickening': 12,
 'Pneumonia': 13,
 'Pneumothorax': 14}

def get_top_disease_info(disease_prob):
    i2l = {i: l for (l, i) in l2i.iteritems()}
    disease_prob_np = np.array(disease_prob)
    top_disease_ids = (-disease_prob_np).argsort()[:5]
    top_diseases = [i2l.get(i) for i in top_disease_ids]
    top_disease_prob = disease_prob_np[top_disease_ids]

    return top_diseases, top_disease_prob


def show_top_disease_cell(ax, top_diseases, top_disease_prob, img_nm):
    y_pos = np.arange(len(top_diseases))

    ax.barh(y_pos, top_disease_prob, align='center',
            color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_diseases)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Disease Probability')
    ax.set_title('Top Diseases - {}'.format(img_nm))

def get_warped_matrix(tgt_img, reference_img):
    # Find homography
    warp_mode = cv2.MOTION_AFFINE  # MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 500;

    # threshold of increment in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(reference_img, tgt_img, warp_matrix, warp_mode, termination_criteria)
    return warp_matrix


def get_warped_img(tgt_img, reference_img, warp_matrix):
    sz = reference_img.shape
    tgt_img = cv2.warpAffine(tgt_img, warp_matrix, (sz[0], sz[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return tgt_img


def get_required_region_mask(img_shape, ignore_border_x_pc, ignore_border_y_pc):
    mask = np.zeros(img_shape)

    img_ht, img_w = img_shape
    ignore_ht = img_ht * ignore_border_y_pc /100
    ignore_width = img_w * ignore_border_x_pc /100

    mask[ignore_ht:(img_ht-ignore_ht), ignore_width:(img_w-ignore_width)] = 1

    return mask

def get_contours_from_bin(thresh, k_size):
    # Ignore minor differences by opening small spots (which represent differences)
    kernel = np.ones((k_size, k_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # finding contours to obtain the regions of the two input images that differ
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    return contours

def get_structural_difference_info(img1, img2, k_size, ignore_border_x_pc, ignore_border_y_pc, diff_threshold):
    # compute Structural Similarity Index (SSIM)
    (score, diff) = compare_ssim(img1, img2, full=True)
    mask = get_required_region_mask(img1.shape, ignore_border_x_pc, ignore_border_y_pc )
    diff = ((1 - diff) * mask * 255).astype("uint8")

    t = (1 - diff_threshold) * 255
    thresh = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("Threshold 1", thresh); cv2.waitKey(0)
    contours = get_contours_from_bin(thresh, k_size)
    return contours, diff, thresh


def get_annotated_image(img1, contours):
    if len(img1.shape) == 2:
        img1_annotated = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_annotated = copy.deepcopy(img1)

    for c in contours:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img1_annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img1_annotated


def get_annotated_image2(img1, contours1, contours2, is_first=True):
    if len(img1.shape) == 2:
        img1_annotated = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_annotated = copy.deepcopy(img1)

    for c in contours1:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        if is_first:
            cv2.rectangle(img1_annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(img1_annotated, (x, y), (x + w, y + h), (0, 0, 0), 2)

    for c in contours2:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)

        if is_first:
            cv2.rectangle(img1_annotated, (x, y), (x + w, y + h), (0, 0, 0), 2)
        else:
            cv2.rectangle(img1_annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img1_annotated


def display_images(img1, img2, img1_annotated, img2_annotated, diff, thresh, img1_name, img2_name, model_path):

    f, axarr = plt.subplots(3,2)
    f.set_size_inches(10, 10)
    f.suptitle("Comparison of {0} (Image 1) and {1} (Image 2) using model:{3}".format(img1_name,img2_name, model_path), fontsize=14)

    axarr[0,0].imshow(img1, cmap='gray')
    axarr[0,0].set_title("Image 1")

    axarr[0,1].imshow(img2, cmap='gray')
    axarr[0,1].set_title("Image 2 (Transformed)")

    axarr[1,0].imshow(img1_annotated)
    axarr[1,0].set_title("Image 1 (Difference highlighted)")

    axarr[1,1].imshow(img2_annotated)
    axarr[1,1].set_title("Image 2 (Difference highlighted)")

    axarr[2,0].imshow(diff, cmap='gray')
    axarr[2,0].set_title("Difference")

    axarr[2,1].imshow(thresh, cmap='gray')
    axarr[2,1].set_title("Threshold")

    f.savefig("ImageComparison.jpg", dpi = 100)
    plt.show()


def imshow_bgr(axarr_i, img):
    axarr_i.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main(options):
    img_dir = options.image_dir
    img1_path, img2_path = img_dir + options.img1, img_dir + options.img2
    side_len = options.output_img_side_length
    model_path = options.model_path

    #img_dir = "/Users/vishalrao/Downloads/images/"
    #img1_path, img2_path = img_dir + "00000001_000.png", img_dir + "00000001_001.png"
    #img1_path, img2_path = img_dir + "00000071_008.png", img_dir + "00000071_004.png"
    #img1_path, img2_path = img_dir + "00000211_019.png", img_dir + "00000211_041.png"
    #img1_path, img2_path = img_dir + "00000211_010.png", img_dir + "00000211_016.png"
    #img1_path, img2_path = img_dir + "00000468_041.png", img_dir + "00000468_017.png"
    img1_path, img2_path = img_dir + "00000090_003.png", img_dir + "00000090_004.png"
    # Read images
    input_img1 = cv2.imread(img1_path, 0)
    input_img2 = cv2.imread(img2_path, 0)

    img1, img2 = copy.deepcopy(input_img1), copy.deepcopy(input_img2)

    # Resize images
    input_img_side_len = img2.shape[0]
    if input_img_side_len != side_len:
        img1 = cv2.resize(img1, (side_len, side_len), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (side_len, side_len), interpolation=cv2.INTER_CUBIC)

    ## Pre-processing
    # Normalize
    img1 = cv2.normalize(img1, img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = cv2.normalize(img2, img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    warp_matrix = get_warped_matrix(img2, img1)

    ## Structural Difference
    # Get warped image
    img2 = get_warped_img(img2, img1, warp_matrix)

    # Get feature differences
    k_size = options.opening_kernel_size
    contours_str, diff_str, thresh_str = get_structural_difference_info(img1, img2, k_size,
                                                            options.ignore_border_x_pc, options.ignore_border_y_pc,
                                                            options.diff_threshold)

    # Annotate images
    img1 = (255*cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
    img2 = (255*cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
    img1_annotated_str = get_annotated_image(img1, contours_str)
    img2_annotated_str = get_annotated_image(img2, contours_str)

    ## Heat Maps
    hm_gen = HeatMapGenerator()
    hm_gen.initialize_with_path(model_path)
    batch_heatmaps, batch_grayscale_heatmaps, batch_disease_probs = hm_gen.generate_heatmap_for_imgPaths([img1_path, img2_path],
                                                                                    annotate_image=False, heatmap_blend_ratio=1.0)
    top5_diseases1, top5_diseases_prob1 = get_top_disease_info(batch_disease_probs[0])
    top5_diseases2, top5_diseases_prob2 = get_top_disease_info(batch_disease_probs[1])

    hm1, hm2 = batch_heatmaps[0], batch_heatmaps[1]
    hm2_warped = get_warped_img(hm2, img1, warp_matrix)

    hm1_gray, hm2_gray = batch_grayscale_heatmaps[0], batch_grayscale_heatmaps[1]
    hm2_warped_gray = get_warped_img(hm2_gray, img1, warp_matrix)

    # Get feature differences
    max_heat_grad_diff_for_sim = options.max_heat_grad_diff_for_sim
    heatmap_threshold = options.heatmap_threshold
    #contours_heat, hm_diff_bin = get_heatmap_difference_info(hm1_gray, hm2_warped_gray, heatmap_threshold, max_heat_grad_diff_for_sim, k_size)
    hm1_t = np.zeros(hm1_gray.shape)
    hm1_t[np.where(hm1_gray > heatmap_threshold)] = hm1_gray[np.where(hm1_gray > heatmap_threshold)]
    """cv2.imshow("hm1", hm1_t)
    cv2.waitKey(0)"""

    hm2_t = np.zeros(hm2_warped_gray.shape)
    hm2_t[np.where(hm2_warped_gray > heatmap_threshold)] = hm2_warped_gray[np.where(hm2_warped_gray > heatmap_threshold)]
    """cv2.imshow("hm2_t", hm2_t)
    cv2.waitKey(0)"""

    diff_pts1 = np.where((hm1_t - hm2_t) > max_heat_grad_diff_for_sim)
    hm_diff_bin1 = np.zeros(hm1_t.shape).astype(np.uint8)
    hm_diff_bin1[diff_pts1] = 255

    diff_pts2 = np.where((hm2_t - hm1_t) > max_heat_grad_diff_for_sim)
    hm_diff_bin2 = np.zeros(hm1_t.shape).astype(np.uint8)
    hm_diff_bin2[diff_pts2] = 255
    """
    cv2.imshow("hm_diff_bin", hm_diff_bin)
    cv2.waitKey(0)
    """
    """
    plt.imshow(hm1_t, cmap="gray")
    plt.show()
    plt.imshow(hm2_t, cmap="gray")
    plt.show()
    plt.imshow(hm_diff_bin, cmap = "gray")
    plt.show()
    """

    diff_contours_heat1 = get_contours_from_bin(hm_diff_bin1, k_size)
    diff_contours_heat2 = get_contours_from_bin(hm_diff_bin2, k_size)

    # Blend heatmaps with input image
    heatmap_blend_ratio = options.heatmap_blend_ratio
    hm1 = cv2.addWeighted(hm1, heatmap_blend_ratio, img1, (1 - heatmap_blend_ratio), 0)
    hm2_warped = cv2.addWeighted(hm2_warped, heatmap_blend_ratio, img2, (1 - heatmap_blend_ratio), 0)
    hm1_annotated = hm_gen.annotate_img(hm1, hm1_gray)
    hm2_annotated = hm_gen.annotate_img(hm2_warped, hm2_warped_gray)

    # Heatmap difference
    hm1_diff = get_annotated_image2(hm1, diff_contours_heat1, diff_contours_heat2, True)
    hm2_diff = get_annotated_image2(hm2_warped, diff_contours_heat2, diff_contours_heat1, False)

    # Display images
    #display_images(img1, img2, img1_annotated_str, img2_annotated_str, diff_str, thresh_str,
    #               os.path.basename(img1_path), os.path.basename(img2_path))

    # Show summary
    os.path.basename(img1_path)
    model_name = os.path.basename(model_path)
    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)

    # Display Vertical
    #f, axarr = plt.subplots(2, 5)

    # Display horizontal
    f, axarr = plt.subplots(3, 4)

    f.set_size_inches(10, 10)
    #f.suptitle("Comparison of {0} (Image 1) and {1} (Image 2) using model:{2}".format(img1_name, img2_name, model_name),
    f.suptitle("Comparison of {0} (Image 1) and {1} (Image 2)".format(img1_name, img2_name),fontsize=14)

    axarr[0, 0].imshow(img1, cmap='gray')
    axarr[0, 0].set_title("Image 1")
    axarr[0, 0].axis('off')

    axarr[1, 0].imshow(img2, cmap='gray')
    axarr[1, 0].set_title("Image 2 (Transformed)")
    axarr[1, 0].axis('off')

    imshow_bgr(axarr[0, 1],hm1_annotated)
    axarr[0, 1].set_title("Probable Affected Area\n(Image 1)")
    axarr[0, 1].axis('off')

    imshow_bgr(axarr[1, 1],hm2_annotated)
    axarr[1, 1].set_title("Probable Affected Area (Image 2)")
    axarr[1, 1].axis('off')

    imshow_bgr(axarr[0, 2],hm1_diff)
    axarr[0, 2].set_title("Heat Map Difference\n(Image 1)")
    axarr[0, 2].axis('off')

    imshow_bgr(axarr[1, 2],hm2_diff)
    axarr[1, 2].set_title("Heat Map Difference (Image 2)")
    axarr[1, 2].axis('off')
    """
    axarr[2, 2].imshow(hm1_t)
    axarr[2, 2].set_title("Heat Map 1\n(Thresholded")

    axarr[2, 3].imshow(hm2_t)
    axarr[2, 3].set_title("Heat Map 2 (Thresholded")

    axarr[3, 2].imshow(hm_diff_bin)
    axarr[3, 2].set_title("Heat Maps Difference Binary")
    """
    imshow_bgr(axarr[0, 3],img1_annotated_str)
    axarr[0, 3].set_title("Structural Difference\n(Image 1)")
    axarr[0, 3].axis('off')

    imshow_bgr(axarr[1, 3],img2_annotated_str)
    axarr[1, 3].set_title("Structural Difference (Image 2)")
    axarr[1, 3].axis('off')

    topDiseaseCell1 = axarr[2, 0]
    show_top_disease_cell(topDiseaseCell1, top5_diseases1, top5_diseases_prob1, "Image 1")

    axarr[2, 1].axis('off')

    topDiseaseCell2 = axarr[2, 2]
    show_top_disease_cell(topDiseaseCell2, top5_diseases2, top5_diseases_prob2, "Image 2")

    axarr[2, 3].axis('off')

    f.savefig("Comparison of images {0} and {1} using model {2}.jpg".format(img1_name, img2_name, model_name), dpi=100)
    plt.show()


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)