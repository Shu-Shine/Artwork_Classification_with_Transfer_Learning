# Fragrant Spaces Classification with Transfer Learning
## Introduction
Olfaction and its aesthetic potential have traditionally been neglected and devalued, given the prevailing emphasis on visual-centric perspectives. It is inevitably absent as a sensory experience evoked through visual mediums such as artworks. Exploring the extraction of scent-related cues from visual data is a significant challenge, an effective way is to leverage the detection of proxies,  such as fragrant spaces, which refer to the scenes within artworks where fragrances or scents are depicted or implied.  

This study presents the first attempt in identifying scenes depicting olfactory experiences within historical artworks using transfer learning techniques. We introduce two novel artistic scene-centric datasets, RASD and WASD, constructed through automated collection and annotation from open cultural heritage data sources. Four state-of-the-art deep neural network architectures, pre-trained on the large-scale photographic dataset, are fine-tuned on these datasets to classify fragrant spaces. The evaluation results demonstrate significant performance improvements, underscoring the efficacy of transfer learning in this context. This research lays a foundation for further exploration into the identification and interpretation of olfactory experiences in cultural heritage.

## Dataset
We possess the artistic scene-centric datasets RASD and WASD for transfer learning, which help bridge the gap between the contrasting characteristics of scent-based artistic representations and physical environments. The Fragrant-Spaces dataset is utilized to assess the efficacy of the trained models, serving as our target set. 
<div align=center>
<img width='500' src='https://github.com/Shu-Shine/Fragrant_Spaces_Classification_with_Transfer_Learning/blob/main/images/t1.jpg'/>
<img width='550' src='https://github.com/Shu-Shine/Fragrant_Spaces_Classification_with_Transfer_Learning/blob/main/images/f.jpg'/>
<img width='550' src='https://github.com/Shu-Shine/Fragrant_Spaces_Classification_with_Transfer_Learning/blob/main/images/asd.jpg'/>
</div>

## Results
Through the application of transfer learning, the assessment of scene classification in artistic fragrant spaces has seen a significant enhancement. The fine-tuned models showcase an average increase of 30.15% in Top-5 accuracy and 18.09% in Top-1 accuracy.
