# Artwork Classification (with transfer learning)

The aim is to classify historical artworks according to a set of fragrant spaces / scenes with a connection to olfaction.

The main idea is to pre-train a model on a photographic dataset for scene classification (places365, http://places2.csail.mit.edu/) and apply it to artistic data relevant to the project.

To bridge the gap between photographic training data and artwork target data, we might have to experiment with domain adaptation techniques, e.g. fine-tuning with a small subset of the artworks or applying style transfer to the training data.
