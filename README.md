# GRL

The code is built under Python 3.11 and Pytorch 2.2.1

###### *Dataset*

|       | Jazz | Cora-ML | Network Science | Twitch Gamers |
| ----- | ---- | ------- | --------------- | ------------- |
| Nodes | 198  | 1098    | 1565            | 1912          |
| Edges | 2742 | 7981    | 13532           | 31299         |

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
- gnn-s.py: estimate the IS using GNN
- dqn.py: train the RL and test
- compare.py: test the performance of the baseline methods
