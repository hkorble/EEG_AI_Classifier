Om Bhatt and I have worked to create a classification model that predicts whether test subjects were watching a happy, sad, or neutral movie scene based on their </b></em>electroencephalogram</em></b> (EEG) brain readings.

Initially, we tried to take this tabular dataset and convert it into an image with a <em><b>Deconvolutional Neural Network</b></em> (De-CNN), followed by classification with a <b><em>Visual Transformer</b></em> (ViT). Unfortunately, it became apparent that we did not have the computational resources for such an endeavor.

We then switched the ViT stage to a pre-trained <b><em>Convolutional Neural Network</b></em> (CNN) and managed to achieve a more reasonable runtime with this De-CNN -> CNN strategy.
