# MaritimeDS

In this study, we aim to construct maritime traffic structure using an unsupervised CycleGAN-based method.

The code has two parts: Generate Map and T2I-CycleGAN. The Generate Map part is to process the raw trajectory data into the input data of the T2I-CycleGAN model; the T2I-CycleGAN part is to process the input data to achieve the extraction of the ocean traffic structure, and the code is based on This part of the code is based on the [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) code to modify the results. The modifications are mainly in the model part of the code, and a sparse layer is added to the CycleGAN code for processing sparse data.

## Generate Map

Code in this section is written using jupyter notebook and can be processed directly according to the steps within the code.

## T2I-CycleGAN

Runtime environment: torch>=1.4.0; torchvision>=0.5.0; dominate>=2.4.0; visdom>=0.1.8.8.

*Run steps:*

1. Create a new dataset storage directory point2region in the `\T2I-CycleGAN\pytorch-CycleGAN\dataset\` , and then create testA, testB, trainA, trainB in point2region. respectively, and testB and trainB store the splitted centerline images divided by 04Generate_Samples_forGAN;

2. Use `python train.py --dataroot . /datasets/point2region --name point2region --model cycle_gan` to train the model;

3. Use `python test.py --dataroot datasets/point2region/mirror_flip180 --name point2region --model test --no_dropout --num_test 256` to perform the test.

## Datasets

* Sample data of Maritime Traffic Structure and Urban Traffic Structure is in `\T2I-CycleGAN\pytorch-CycleGAN\dataset\point2region\`

* The benchmark data used in *Urban Traffic Structure* is[OpenStreetMap (OSM)](https://www.openstreetmap.org/).
  
