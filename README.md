Fast Style Transfer (Johnson et al, 2016)
=========================================
This repository contains a simple implementation of Johnson's style transfer
model in PyTorch.  For a detailed description, have a look at my repository
[`johnson-fast-style-transfer`](https://github.com/mdehling/johnson-fast-style-transfer).

> [!NOTE]
> This repository contains a demo notebook.  There are two simple ways to run it
> without installing any software on your local machine:
>
> 1. View the notebook on GitHub and click the _Open in Colab_ button (requires
>    a Google account).
> 2. Create a GitHub Codespace for this repository and run the notebook in
>    VSCode (requires a GitHub account).

Training
--------
The included training script uses the hydra package for easy configuration
management.  To train a style transfer model for a single style image, run
`./train.py style_image=candy.jpg`.  To see available options, run
`./train.py --help`.  To train style transfer models for a range of style
images, run the following command:

    ./train.py --multirun style_image=bathing.jpg,candy.jpg,cubism.jpg,delaunay.jpg,scream.jpg,starry-night.jpg,udnie.jpg

The directory `multirun` contains the results of this exact command.
