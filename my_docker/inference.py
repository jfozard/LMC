"""
The following is a simple template algorithm for the Ligh My Cells Challenge.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from /input and outputs to /output

To export the container and prep it for upload to Grand-Challenge.org you can call:

docker save ${DOCKER_TAG} | gzip -c > ${DOCKER_TAG}.tar.gz

"""
from pathlib import Path
from os.path import join, isdir, basename
from os import mkdir, listdir
import tifffile
import xmltodict
from mynetwork import MyNetWork
import torch
import time




INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
if not isdir(join(OUTPUT_PATH,"images")): mkdir(join(OUTPUT_PATH,"images"))
RESOURCE_PATH = Path("resources")

def run():

    #Load the network
    model = MyNetWork(RESOURCE_PATH / "model_and_optimizer_140.pth", RESOURCE_PATH / "actin_model_and_optimizer_199.pth", RESOURCE_PATH / "tubulin_model_and_optimizer_200.pth" )
#    model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean') 
    
    #List the input files
    transmitted_light_path = join(INPUT_PATH , "images","organelles-transmitted-light-ome-tiff")

    t0 = time.time()
    for input_file_name in listdir(transmitted_light_path):
        if input_file_name.endswith(".tiff"):

            print(" --> Predict " + input_file_name)
            image_input,metadata=read_image(join(transmitted_light_path,input_file_name))

            # Get the type of transmited liht (BF, DIC, PC)
            description = xmltodict.parse(metadata)
            print(description)
            desc = description["OME"]['Image']["Pixels"]["Channel"]
            if '@Name' in desc:
                tl = desc['@Name']
            elif '@Fluor' in desc:
                tl = desc['@Fluor']
            else:
                tl = 'none'
            
            print(input_file_name, tl)

            image_predict = model.forward(image_input, tl)

            for idx, organelle in enumerate(["Nucleus", "Mitochondria", "Tubulin", "Actin"]):
                # Perform the prediction

                #Save your new predicted images
                output_organelle_path = join(OUTPUT_PATH, "images", organelle.lower() + "-fluorescence-ome-tiff")
                if not isdir(output_organelle_path):  mkdir(output_organelle_path)
                save_image(location=join(output_organelle_path,basename(input_file_name)), array=image_predict[idx],metadata=metadata.encode())


    print('time:', time.time() - t0)
    return 0



def read_image(location):
    # Read the TIFF file and get the image and metadata
    with tifffile.TiffFile(location) as tif:
        image_data = tif.asarray()    # Extract image data
        metadata   = tif.ome_metadata # Get the existing metadata in a DICT
    return image_data, metadata


def save_image(*, location, array,metadata):
    #Save each predicted images with the required metadata
    print(" --> save "+str(location))
    pixels = xmltodict.parse(metadata)["OME"]["Image"]["Pixels"]
    physical_size_x = float(pixels["@PhysicalSizeX"])
    physical_size_y = float(pixels["@PhysicalSizeY"])
    
    tifffile.imwrite(location,
                     array,
                     description=metadata,
                     resolution=(physical_size_x, physical_size_y),
                     metadata=pixels,
                     tile=(128, 128),
                     )


if __name__ == "__main__":
    raise SystemExit(run())
