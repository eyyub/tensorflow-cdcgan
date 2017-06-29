# tensorflow-cdcgan
This is a short implementation of a Conditional DCGAN, however if you need a cDCGAN for real-world use cases, please consider using a more serious implementation.

[@eyyub_s](https://twitter.com/eyyub_s)

## Samples
![](https://raw.githubusercontent.com/Eyyub/tensorflow-cdcgan/master/gifs/mnist.gif?token=ABzE8Hr7pVAnwkpsSUFf1jNsdGfAqT5hks5ZXYJvwA%3D%3D)

Here can be seen an interesting property of cGANs: each rows represent a random noise, and each columns represent samples that have been generated using this noise and a class-label.
The generator is able to produce the right kind of digit we asked for using the same noise!

## Fun
![](https://github.com/Eyyub/tensorflow-cdcgan/blob/master/gifs/fail.gif?raw=true)

Here can be seen a cDCGAN trained on CIFAR-10 using the same networks architectures I used for MNIST, obviously it shows that we need to be careful when designing the architecture. It works better using more filters.

## Requirements
I'm a linux guy but I was running on Windows when I did this project.
- Python 3.5
- Tensorflow for Windows
- Matplotlib
