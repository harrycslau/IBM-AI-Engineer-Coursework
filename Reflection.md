I have encountered these problems when completing this project.

-  Disconnected or Kernel Died Unexpectedly
   These two problems occurred many times, as the unzip and training process are particluarly long.

-  Corrupted Files
   Possibly due to the long unzip time and disconnection problem. There are two files failed to read, and I replaced them by copying the nearby file.

-  Omitted the transformation
   I did not realized that the ResNet18 accepts an image size of 224 only, not 512. Ignoring this will cause the error either too large or too small.
