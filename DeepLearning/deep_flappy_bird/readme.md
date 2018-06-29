### Overview

Training an AI for playing flappy bird using keras.

### Comment

I implement a simple replay memory which considers the age of the state. Thus the states with larger T are more likely to keep in the memory as it is less often encountered.

### Dependency 

1. DeepLearningFlappyBird in github: https://github.com/yenchenlin/DeepLearningFlappyBird

Put this single file at the root of that repository.

2. keras

The version of keras I'm using is 1.1.0. As for as I know, the interface changed a lot in the later versions.
