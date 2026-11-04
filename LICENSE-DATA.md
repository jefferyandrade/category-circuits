# Data licence: CC BY 4.0

The data in this repository, and the circuit library when it is deposited, are licensed under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/
(legal code: https://creativecommons.org/licenses/by/4.0/legalcode).

It covers:
- everything under `data/`;
- the category tree and its inputs under `taxonomy/inputs/`;
- the circuit library (every circuit's weight masks, receipts and pruning logs), to be deposited separately.

The code is under the MIT licence (`LICENSE`).

## Attribution
Cite the paper (see `CITATION.cff`):
Andrade, J. W., Vaziri, P. A., Konkle, T., & Alvarez, G. A. (2026). Beyond representations: the wiring of visual
untangling in deep networks carries taxonomic category structure.

## Third-party sources
- The category tree is derived from WordNet 3.0. WordNet 3.0 Copyright 2006 by Princeton University. All rights
  reserved. It is used under the WordNet licence (https://wordnet.princeton.edu/license-and-commercial-use), which
  permits use, copying and distribution provided this notice is kept.
- Category indices, WordNet ids and class names follow the ImageNet 2012 classification label set, as distributed with
  torchvision. No ImageNet images are included.
- The networks are the pretrained torchvision checkpoints (AlexNet, VGG-11, ResNet-18 and ResNet-50, all
  `IMAGENET1K_V1`), which are not redistributed here.
