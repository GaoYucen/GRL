# GRL

The code is built under Python 3.11 and Pytorch 2.2.1

###### *Dataset*

|       | Jazz | Network Science | Twitch Gamers | Digg      |
| ----- | ---- | --------------- | ------------- |-----------|
| Nodes | 198  | 1,565            | 1,912          | 279,613   |
| Edges | 2,742 | 13,532           | 31,299         | 1,170,689 |

###### *Code*

To test the code, simply run the following command  
```
cd <GRL>
python dqn.py
```

###### *Code Structure*

- diffusion_model.py: define the IC propagation model
- baseline_model.py: define the baseline methods
- gnn.py: define the model structure of GNN
- gnn-is.py: estimate the IS using GNN
- dqn.py: train the RL and test
- compare.py: test the performance of the baseline methods
