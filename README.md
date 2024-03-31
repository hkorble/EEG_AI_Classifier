Om Bhatt and I have worked to make a classification model which predicts whether test subjects were watching a happy, sad, or neutral movie scene based on their EEG brain reading.

Initally we tried to take this tabular dataset and convert it into an image with a <em>Deconvolutional Neural Network</em>(De-CNN), followed by classification with a <em>Visual Transformer</em>(ViT). Unfortunaetely it became aparent that we did not have the computational resources for such an endevour.

We then switched the ViT stage to a pre-trained <em>Convolutional Neural Network</em> (CNN) and managed to acheive a more reasonable runtime with this De-CNN -> CNN strategy

