#### Part A

In determining an optimal value for k, the rule-of-thumb approach is to select a value that gives you sufficient visual quality for the lowest value. Therefore, it is subjective by nature and we are to either visually or mathematically apply a threshold. Principal Components (PCs) with lower eigenvalues have smaller effects on the reconstruction, so one approach may be to apply a minimum threshold for eigenvalue (some suggest setting this as 1). Another approach along a similar vein relates to the cumulative proportion of the variance explained by the principal components. Similarly, then can apply a threshold for this cumulative proportion (as this approaches a value of 1 we may select 0.99). Finally, we can consider the similarity of the resulting reconstructed image to the original image using the structural similarity index (SSIM); for image reconstructions with cumulative PCs (see &#39;k=[0].png&#39;, &#39;k=[0,1].png&#39; etc. in output folder) we can use SSIM to compare regions of target pixels between the reconstructed image and original image. See &#39;output/SSIM\_by\_PCs.png&#39;.

e.g.

<p align="center">
  <img src="/partA/output/k=[0].png" width="500">
</p>

<p align="center">
  <img src="/partA/output/k=[0, 1].png" width="500">
</p>

<p align="center">
  ......
</p>

<p align="center">
  <img src="/partA/output/k=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].png" width="500">
</p>


<p align="center">
  <img src="/partA/output/SSIM_by_PCs.png" width="800">
</p>

Demonstrably, both visually (see comparison below) and quantitatively (see SSIM plot above) the point of diminishing returns for image quality appears to be when we use the first 4 PCs (zero indexed: k=[0, 1, 2, 3]).

| Loaded LSP data   |  vs. |  Randomly cropped and rotated |
-------------------------:|:-------------------------:|:-------------------------:|
<img src="/partA/output/k=[0, 1, 2, 3].png" width="500">  |    |  <img src="/partA/output/k=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].png" width="500">|

| First 4 PCs   |      vs.     |  All PCs |
|----------|:-------------:|------:|
| <img src="/partA/output/k=[0, 1, 2, 3].png" width="500"> |  - | <img src="/partA/output/k=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].png" width="500"> |


#### Part B

Data augmentation is the process of applying a variety of transformations to expand a dataset. This has relevance in deep learning in the context of the phenomenon of overfitting, whereby the network learns a function that fits the training dataset well, but fails when presented with in-the-wild examples that are dissimilar to examples in the dataset. This problem can be reduced in a number of ways. One approach is to increase the dataset with new annotated images, however, generating this data is expensive and time consuming. An inxpensive and quick way to reduce overfitting is to augment your training dataset with transformations of existing images. In the script LSP_data_augmentation.ipbyn I perform data augmentation on the Leeds Sports Pose (LSP) dataset. This is one example of a publically available training dataset that is used for 2D human pose estimation, it is specialised for sports poses which are less well represented in other datasets (e.g. Common Objects in Context (COCO)). One limitation of this dataset is that it contains significantly fewer training images than other datsets, i.e. 10,000 in LSP vs. 250,000 in COCO. 
In the notebook script I perform an elementary augmentation of LSP by:
1) Creating a data loader which reads in images and their associated annotated keypoints (which are in matlab format '.mat')
2) plot a random set of images with annotated keypoints overlayed
3) calculate image and keypoint properties; center and scale
4) randomly crop and rotate the the random set of images and annotated keypoints

Note: To run this you must download the LSP extended dataset (available here: https://sam.johnson.io/research/lspet.html) and save to partB/lspet_dataset:
```
CS7GV1_Assignment1
|-- partB
`-- |-- lspet_dataset
    `-- |-- images
        |   |-- im00001.jpg
        |   |-- im00002.jpg
        |   |-- im00003.jpg
        |   |-- ....
        |   |-- im10000.jpg
        `-- visualized
        |   |-- im00001.jpg
        |   |-- im00002.jpg
        |   |-- im00003.jpg
        |   |-- ....
        |   |-- im10000.jpg
        `-- joints.mat
        `-- README.txt

```

e.g.

| Loaded LSP data   |  Randomly cropped and rotated |
|----------|------:|
| <img src="/partB/backup/output_2_0.png" width="300"> | <img src="/partB/backup/output_4_0.png" width="250"> |
| <img src="/partB/backup/output_2_1.png" width="300"> | <img src="/partB/backup/output_4_1.png" width="250"> |
| <img src="/partB/backup/output_2_3.png" width="300"> | <img src="/partB/backup/output_4_3.png" width="250"> |

