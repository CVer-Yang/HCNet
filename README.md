Remote sensing image captioning aims to describe the crucial objects from remote sensing images in the form of natural language. Currently, it is challenging to generate high-quality captions due to the multi-scale targets in remote sensing images and the cross-modality differences between images and text features. To address these problems, this paper presents an approach for generating captions through hierarchical feature aggregation and cross-modality feature alignment, namely HCNet. Specifically, we propose a hierarchical feature aggregation module (HFAM) to obtain a comprehensive representation of vision features. Considering the disparities among different modality features, a cross-modality feature interaction module (CFIM) is designed in the decoder to facilitate feature alignment. Meanwhile, a cross-modality align loss is introduced to realize the alignment of image and text features. Extensive experiments on the three public caption datasets show our HCNet can achieve satisfactory performance. Especially, we demonstrate significant
performance improvements of +14.15\% CIDEr score on NWPU datasets compared to existing approaches.


First, refer to the [MLAT](https://github.com/Chen-Yang-Liu/MLAT) to generate the required data in the data\UCM_images1.

Then, python train_HCNet_UCM.py, generate the weights in the best_UCM_weights.

Finally, python eval_HCNet_UCM.py.

This code is based on the [MLAT](https://github.com/Chen-Yang-Liu/MLAT) and [Clip](https://github.com/openai/CLIP).
