# Tag Embeddings

This folder is dedicated to the module which would get the words embeddings and hence the VA/VAD ratings. 

## Notes
1. Social Tags -> Embeddings (FastText)
2. Words -> VAD (ANEW)
3. Embeddings -> VA (Train a neural network)

### Logs
1. The inittial model has been traoined with the original values -- had resulted in too large of errors. ... The following things were done to reduce the model's loss
   1. Adjusting the learning rate -- from 1e-3 to 1e-6  -- Although some value have looked promising, they would take a lot of epochs to converge to a decent loss. 
   2. It was observed that the input value and the VA values have a scale difference. 
      1. Hence, standardisation scaling had been decided to be done for the Output values. 
      2. Since it is linear, the relations between each of the words such as the nearness and farness have been fine. 
      3. I have tried scaling X to the mean and std of Y, did not change much. (25th Apr - 16:50:52 trail)
      4. But once I have scaled the embeddings and the output, the loss has dropped. (Even if I have not scaled the embeddings.)



## Links
ANEW Dataset Link: [link](https://osf.io/y6g5b/wiki/anew/)
XANEW Dataset Link: [link](https://github.com/JULIELab/XANEW)
