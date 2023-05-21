# sky_segmentation
The subject of this test is to propose a solution in order to automate the creation of segmentation masks for sky in images. Being able to automate this kind of tasks could open up great opportunities for the creation of new automated effects.

We would like you to propose at least two different approaches for this task.
Hint: that can go from clustering or basic binary classification techniques to CNN based solutions. Don’t hesitate to use the specificities of sky to implement your solution.

## Installation
To install all dependencies and set up the Python environment, we recommend using [poetry](https://python-poetry.org/docs/). This will also install the Python package created in this repo, video_recover.

For more information about poetry, visit the official documentation.

To install the dependencies, run the following command:

    poetry install

To enter the environment in the shell, run:

    poetry shell

To verify that everything is working, start a Python interpreter in the command line and run:

    import sky_segmentation

Alternatively, you can use any package manager of your choice. The main dependencies can be found in the [toml file](pyproject.toml) file.

## Solutions

1. **Deep Learning Based**

- The [MIT Scene Parsing Benchmark](https://github.com/CSAILVision/sceneparsing) provides a standard training and evaluation platform for the algorithms of scene parsing. Scene parsing is the task of segmenting and parsing an image into different image regions associated with semantic categories, such as sky, road, person, and bed.

- There are in total 150 semantic categories included for evaluation, which include e.g. sky, road, grass, and discrete objects like person, car, bed. However, in this problem we are interested in the `sky` class.

- The dataset can be downloaded:

        ./bash/download.sh


    1.1. **Use a Pretrained Model**

- The simplest solution is use a pretrained model on this dataset and implement a wrapper to obtain the segmentation mask for the `sky` class only.


- The publishers of the dataset have also open-sourced the weights for pretrained models that can be found [here](https://github.com/CSAILVision/semantic-segmentation-pytorch).

- The wrapper model is `SkyPretrainedSegmentationModule` and implemented [here](sky_segmentation/modules/models.py). This [notebook](notebooks/pretrained_deep.ipynb) showcase using it for inference.

- You need to download the weights first:

        ./bash/download_ckpt.sh


    1.2. **Train a Custom Model**

-  As we are interested in the `sky` only, it would be better to train a custom model that produces a binary mask to indicate the sky regions in a given image. We expect this task to be simpler to solve for the model compared to the original dataset.

- The main script that runs the training is located [here](scripts/train.py). To run it, execute the following command:

        python scripts/train.py

- If you have a slurm cluster you can submit the training job:

        sbatch bash/train.sh

- The train script filter the images that contains the `sky` images first, then trains a `UNet` model with `VGG18` as an encoder (pretrained on ImageNet)

- To test the trained model:

        python scripts/test.py


2. **Classical Image Processing Based**

    2.1. **Contour-Based Segmentation**

- This solution is based on a series of image processing steps. It starts by converting the image to binary image. Then, a contour algorithm is applied to find the biggest contour that should contain everything except the sky. Some Morpholigical operations are applied to refine the segmentation mask.

- The implementation can be found in the `ContourBasedSkySegmentation` class [here](sky_segmentation/image_processing/segmentation.py), whereas the notebook that showcases it can be found [here](notebooks/classic_contour.ipynb)


    2.2. **Color-Based Segmentation**

- This solution is based on the intrinsic property of the sky: the color. First, it converts the image to HSV color space. Then, the image is thresholded based on the range of blue-sky color.

- The implementation can be found in the `ColorBasedSkySegmentation` class [here](sky_segmentation/image_processing/segmentation.py), , whereas the notebook that showcases it can be found [here](notebooks/classic_color.ipynb)
