# Cell segmentation self-organizing maps

Sort nucleis in H&E slides according to their shapes.

## Usage

For testing on the example slide from `input` directory:
1. Clone this repository
2. Install `requirements.txt` for running example analysis notebook
3. Run the `segsom-example-notebook.ipynb` notebook

For running analysis on your own image:
1. Segment the HE using https://hub.docker.com/r/kidzik/he_cell_seg (or if you have NVIDIA GPU, use a GPU-accelerated version https://hub.docker.com/r/kidzik/he_cell_seg_gpu)
2. Update file lecation and segmentation mask location in the notebook
3. Run the `segsom-example-notebook.ipynb` notebook

## Credits

Code in this repository by Edwin Yuan and Magdalena Matusiak.

Cell segmentation scripts and the neural network by Korsuk Sirinukunwattana.
