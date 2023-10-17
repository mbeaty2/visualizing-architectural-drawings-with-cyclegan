# Artificial-Intelligence-for-the-Media-Mini-Project

The files found in this repository are all files related to my Mini-Project for the Artificial Intelligence for the Media course. The only files not included in this repository are the datasets, as they are too large to be uploaded. These datasets are available via sharepoint here: https://artslondon-my.sharepoint.com/:f:/g/personal/m_beaty0520221_arts_ac_uk/EoGmv2Sotb1KlRN0t10Oyn4BNAiE2HVdAkegYp37dzhEiQ?e=b5tG3h.

The rest of this repository includes:
1. Four notebooks attempting various Pix2Pix models. These notebooks use the Image Pairs and Training_small datasets available on sharepoint. Both of these datasets are within the Pix2Pix datasets folder. 

2. One notebook attempting a CycleGAN generative model. This notebook uses the CycleGAN dataset available on the sharepoint.

3. A folder containing the CycleGAN generated images.

4. A PDF Project Brief explaining the process involved with this project, the project aims, difficulties, and results. 

# Visualizing Architectural Drawings with CycleGAN

This project aims to develop a generative AI capable of producing realistic architectural images from building outlines and sketches. 

## Description

This project was developed in part of the Artificial Intelligence for the Media course completed as part of my Masters of Data Science and AI. This project was interested in automating the process of visualizing architectural drawings as a means of speeding up the process from conception to visualization and ultimately to build of architectural spaces. It looks at using a CycleGAN and Pix2Pix model trained on a self-generated set of image pairs. The image pairs were created by acquiring a dataset of images capturing images, and running those images through a CANNY edge detection model to get their outlines. To fit the hand-in specifications, the project was originally written and tested within a jupyter notebook that has been uploaded to this repository. There are several notebooks testing various different models. Specifically, four notebooks that test various implementations of a Pix2Pix model. All of these model's failed to produce successful results. A jupyter notebook, and python file, containing a CycleGAN model was the only successful model trained for this project. The CycleGAN model was built and adapted from that within this article: https://www.kaggle.com/code/songseungwon/cyclegan-tutorial-from-scratch-monet-to-photo

## Getting Started

### Dependencies

* The foundation of this project is within pytorch.
* This project utilized a 16GB GPU.
* Other dependencies can be found within the CycleGAN jupyter notebook or the python file. 

### Installing

* Create a dataset of image pairs.
* Download and implement the CycleGAN jupyter notebook or python file. The other models are also available for those interested in getting them working. 

### Executing program

* Download the necessary datasets. 

If using the Jupyter Notebook:
* Open the file in a code editor and connect to a code environment.
* Run each cell in the notebook. Should you want to make adjustments, be sure to save the file before continuing to run each cell. If there are issues, restart the kernel and re-run each cell.

If using the python files:
* Download the .py files
* Open a terminal and locate where the downloaded datasets and files are on your local machine.
* Create a new environment to run everything in.
* Run this line in your terminal: 

## Help

The most common problem I encountered was run-time lengths and compatibility errors with the Pix2Pix model infrastructure with my local machine. I encourage those interested in pursuing this project to ensure they have a large enough GPU to handle the complex models, as well as to turn more towards the CycleGAN model than the Pix2Pix model, as the CycleGAN model was far more successfully trained than the Pix2Pix model.  

## Authors

Marissa Beaty, https://github.com/mbeaty2

## Version History
* 0.1
    * Initial Release

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
