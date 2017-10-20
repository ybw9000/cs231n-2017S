Solutions to Stanford cs231n-2017S

--In assignment 2, using stride_tricks and tensordot function of numpy to implement fast conv and pooling layer can achive 
  10% faster than the provided conv layer in forward pass and 2X faster in the backward pass of the pooling layer. Check 
  conv_forward_tensordot function for details.
  
--In assignment 3, instead of generating digits from [0-9] at the same time with a GAN, generating/training a particular 
  digit once a time gives more real images. See GANs-classifier.ipynb for details.
