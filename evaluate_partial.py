"""
The following is the evaluation method for the France-BioImaging Light My Cells Challenge.

It is meant to run within a container.
This will start the evaluation, reads from ./test/input and outputs to ./test/output
To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-evaluation-phase-1 | gzip -c > example-evaluation-phase-1.tar.gz
"""
import json
from multiprocessing import Pool
from os.path import isfile
from pathlib import Path
from pprint import pformat, pprint
import os
import numpy as np
from scipy.stats import pearsonr
from tifffile import imread
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import pandas as pd

INPUT_DIRECTORY = Path("input")
print(" INPUT_PATH IS " + str(INPUT_DIRECTORY))
os.system("ls -lh   " + str(INPUT_DIRECTORY))

OUTPUT_DIRECTORY = Path("output")
print(" OUTPUT IS " + str(OUTPUT_DIRECTORY))
os.system("ls -lh " + str(OUTPUT_DIRECTORY))

GROUND_TRUTH_DIRECTORY = Path("ground_truth")
print(" GROUND_TRUTH_DIRECTORY IS  " + str(GROUND_TRUTH_DIRECTORY))
os.system("ls -lh " + str(GROUND_TRUTH_DIRECTORY))


def get_img_metrics(organelle, img_gt, img_pred):
    '''
    Calculate image metrics for a given organelle.

    Parameters:
    - organelle (str): The type of organelle.
    - img_gt (np.ndarray): Ground truth image.
    - img_pred (np.ndarray): Predicted image.

    Returns:
    Dict[str, float]: Dictionary containing calculated metrics
    for the wanted organnelle.
    '''

    metrics_results = {}

    if img_gt is None :
        print("Image GT is None")
        return metrics_results

    if img_pred is None:
        raise ValueError("ERROR None prediction image")

    if img_gt.shape != img_pred.shape: # NOT SAME SHAPE
        print(f"{organelle} : ERROR SHAPES are not equal ! GT shape = {img_gt.shape} and Pred shape = {img_pred.shape}")
        raise ValueError(f"ERROR GT and predictions shapes are not equal ! \n \
                           GT shape = {img_gt.shape} and Pred shape = {img_pred.shape}")

    # perform percentile normalization
    try:
        img_gt = percentile_normalization(img_gt)
    except Exception as e:
        print(" --> ERROR NORMALIZATION GT ")
        print(e)
        return metrics_results

    try:
        img_pred = percentile_normalization(img_pred)
    except Exception as e:
        print(" --> ERROR NORMALIZATION PRED ")
        print(e)
        return metrics_results

    PCC, _ = pearsonr(img_gt.ravel(), img_pred.ravel() )

    SSIM = ssim(img_gt,
                img_pred,
                data_range=np.maximum(img_pred.max() - img_pred.min(),
                                      img_gt.max() - img_gt.min()))

    metrics_results['PCC'] = 0 if np.isnan(PCC) else PCC
    metrics_results['SSIM'] = SSIM

    if any(x in organelle for x in ["nucleus", "mitochondria"]):

        MAE = mean_absolute_error(img_gt, img_pred)
        e_dist = euclidean_distances(img_gt.ravel().reshape(1, -1),
                                     img_pred.ravel().reshape(1, -1))[0, 0]
        c_dist = np.abs(cosine_distances(img_gt.ravel().reshape(1, -1),
                                         img_pred.ravel().reshape(1, -1))[0, 0])

        metrics_results['MAE'] = MAE
        metrics_results['E_dist'] = e_dist
        metrics_results['C_dist'] = c_dist

    print(metrics_results)
    return metrics_results



metrics_methods = {}
metrics_methods["nucleus"] = ['MAE', 'PCC', 'SSIM', 'E_dist', 'C_dist']
metrics_methods["mitochondria"] = ['MAE', 'PCC', 'SSIM', 'E_dist', 'C_dist']
metrics_methods["actin"] = ['PCC', 'SSIM']
metrics_methods["tubulin"] = ['PCC', 'SSIM']


def main():
    '''
    Main function for the evaluation process.
    It processes each algorithm (=job) for this submission.
    then it reads the predictions.json (in the input folder) and
     starts a number of process workers (using multiprocessing)
    to compute the metrics between the prediction and ground truth images.
    Finally, it writes the metrics of each image into a json file (output folder).
    '''
    print(" START ")
    print_inputs()

    input_files = [x.name for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print(input_files)
    
    metrics = {}
    print("READ PREDICTIONS")

    #predictions = read_predictions()

    predictions = []
    for f in input_files:
        predictions.append({'input_filename':f})

    gt_path = OUTPUT_DIRECTORY / "images"

    
    with Pool(processes=4) as pool:
        metrics["results"] = pool.map(process, predictions)

    metrics['aggregate'] = calcul_mean_organelles(metrics)

    print(metrics['aggregate'])

    
    print("METRICS OK ")
    # # Make sure to save the metrics
    write_metrics(metrics=metrics)

    print("METRICS written")
    return 0


def process(job):
    """
    Process a single algorithm job.

    Parameters:
    - job (Dict[str, Any]): Job information.

    Returns:
    Dict[str, Union[float, Dict]]: Results of the processing.
    """

    input_filename = str(job["input_filename"])
    print(input_filename + " -> Processing:")


    print(input_filename + " -> LOAD PRED IMAGES")
    # Load the predictions

    output_path = OUTPUT_DIRECTORY / "images"
    
    nucleus_pred = load_image_file(filename=output_path / "nucleus-fluorescence-ome-tiff" / input_filename)
    mitochondria_pred = load_image_file(filename=output_path / "mitochondria-fluorescence-ome-tiff" / input_filename)
    tubulin_pred = load_image_file(filename=output_path / "tubulin-fluorescence-ome-tiff" / input_filename)
    actin_pred = load_image_file(filename=output_path / "actin-fluorescence-ome-tiff" / input_filename)

    print(input_filename + " -> CHECK if NO missing pred")
    # check if there is no missing prediction
    if any(x is None for x in [nucleus_pred, mitochondria_pred, tubulin_pred, actin_pred]):
        raise ValueError(job_pk + " ->  ERROR: MISSING PREDCTIONS FILES ")

    # Load and read the ground truth
    gt_path = GROUND_TRUTH_DIRECTORY /"images"
    print(input_filename + " -> LOAD GT")

    input_base = 'image_'+input_filename.split('_')[1]
    nucleus_gt = load_image_file(filename=gt_path / (input_base + '_Nucleus.ome.tiff'))
    mitochondria_gt = load_image_file(filename=gt_path / (input_base + '_Mitochondria.ome.tiff'))
    tubulin_gt = load_image_file(filename=gt_path / (input_base + '_Tubulin.ome.tiff'))
    actin_gt = load_image_file(filename=gt_path / (input_base + '_Actin.ome.tiff'))

    # Calculate and group the metrics by comparing the ground truth to the actual results
    results = {}
    results["image_name"] = input_filename
    results["nucleus"] = get_img_metrics("nucleus", nucleus_gt, nucleus_pred)
    results["mitochondria"] = get_img_metrics("mitochondria", mitochondria_gt, mitochondria_pred)
    results["tubulin"] = get_img_metrics("tubulin", tubulin_gt, tubulin_pred)
    results["actin"] = get_img_metrics("actin", actin_gt, actin_pred)

    
    print(input_filename + " -> METRICS "+str(results))
    return results




def load_image_file(*,  filename):
    """
    Load an image file.

    Parameters:
    - filename (Path): Path to the image file.

    Returns:
    Optional[np.ndarray]: Loaded image as a NumPy array, or None if the file is not found.
    """
    if not isfile(filename):
        print(" MISS "+str(filename))
        return None

    print(" FOUND "+str(filename))
    image = imread(filename)
    return image


def percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    '''
    Compute a percentile normalization for the given image.

    Parameters:
    - image (array): array of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute. 
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed. 
             The default (=None) is to compute it along a flattened version of the array.
    - dtype (dtype): type of the wanted percentiles (uint16 by default)

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    '''

    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100 ):
        raise ValueError("Invalid values for pmin and pmax")

    low_p = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_p = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_p == high_p:
        img_norm = image
        print(f"Same min {low_p} and high {high_p}, image may be empty")
    else:
        img_norm = (image - low_p) / (high_p - low_p)

    return img_norm


def print_inputs():
    """
    Just for convenience,
    Prints information about input files in the logs.

    Returns:
    None
    """
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def calcul_mean_organelles(metrics):

    mean_organelle={}
    for elet in metrics['results'] :
        for organelle in ["nucleus", "mitochondria", "actin", "tubulin"]:
            if organelle in elet and len(elet[organelle]) > 0:
                # not 'actin': {}
                if organelle not in mean_organelle:
                    mean_organelle[organelle] = {}

                for method in metrics_methods[organelle]:
                    if method not in mean_organelle[organelle]:
                        mean_organelle[organelle][method] = []
                    mean_organelle[organelle][method].append(elet[organelle][method])

    metrics_mean = {}
    for organelle in mean_organelle:
        metrics_mean[organelle] = {}
        for method in mean_organelle[organelle]:
            print(organelle, method, mean_organelle[organelle][method])
            metrics_mean[organelle][method] = np.mean(mean_organelle[organelle][method])

    return metrics_mean


    
def write_metrics(*, metrics):
    """
    Writes a JSON document used for ranking results on the leaderboard.

    Parameters:
    - metrics (dict): Metrics to be written to the document.

    Returns:
    None
    """
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))

    print(" CHECK OUTPUT IS   "+str(OUTPUT_DIRECTORY))
    os.system("ls -l " + str(OUTPUT_DIRECTORY))


if __name__ == "__main__":
    raise SystemExit(main())
