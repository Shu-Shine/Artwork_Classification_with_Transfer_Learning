# Novel Artistic Scene-Centric Datasets for Effective Transfer Learning in Fragrant Spaces
## Introduction
Olfaction is often overlooked in cultural heritage studies, while examining historical depictions of olfactory scenes can offer valuable insights into the role of smells in history. The main challenge arises from the lack of published datasets with scene annotations for historical artworks, especially in artistic fragrant spaces. We introduce a novel artistic scene-centric dataset consisting of 4541 artworks and categorized across 170 distinct physical scene categories. We show that a transfer learning approach using weakly labeled training data can remarkably improve the classification of fragrant spaces and, more generally, artistic scene depictions. This work lays a foundation for further exploration of olfactory spaces recognition and broadens the classification of physical scenes to the realm of fine art. All images and labels are released as the ArtPlaces dataset at https://zenodo.org/records/13371280.

## ArtPlaces Dataset
We create two source datasets by retrieving images from the Rijksmuseum collection and Wikidata. The respective query terms are used as weak (i.e. semi-automatically generated) labels which serve as supervision signals during fine-tuning. Additionally, we create a manually labeled dataset of olfaction-related artworks to test the algorithms’ capability to classify fragrant spaces. 
We combine the three source datasets to derive the ArtPlaces dataset:
1. ArtPlaces-train: Is the weakly labeled training split, obtained by combining parts of RASD and WASD.
2. FragrantSpaces-test: The primary objective of the FragrantSpaces-test set is to evaluate the models’ ability to detect fragrant spaces in olfaction-related artworks. As it is based on the ODOR dataset [32], we can assume that all of the images have some relation to olfaction. This focused approach allows us to measure how well the models identify and classify fragrant environments, which is crucial for their application in automated smell-reference extraction.
3. ArtPlaces-test: In contrast, the ArtPlaces-test split is designed to provide a broader evaluation of scene classification capabilities. It does not specifically focus on olfaction-related images but aims to assess the general scene classification capabilities. This dataset offers a robust framework for testing scene recognition performance at the expense of a specific focus on smell-related scenes.




