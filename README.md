## Usage:

### Load Model

```python
BPMs = ['BDS_BTS:BPM_D5513', 'BDS_BTS:BPM_D5565']
models = {}

for bpm in BPMs:
    fname = bpm.replace('_D', '')[-7:]
    state_dict = torch.load(fname + '/model.pt')
    model_info = {
        'n_node': len(state_dict['nn.2.bias']),
        'n_hidden_layer': int((len(state_dict) - 2) / 2) - 2,
        'dtype': state_dict['nn.2.bias'].dtype
    }
    model = BPMQ_model(**model_info)
    model.load_state_dict(state_dict)
    models[bpm] = model

### BPMQ Prediction from 4 Pickup Data

#### Prepare Input Data

```python
for bpm in BPMs:
    # 'Ui' is calibrated TIS161_i signal, i.e., Ui = TIS161_i * TIS161_i_coeff
    U[bpm] = torch.tensor(df[bpm][['U1', 'U2', 'U3', 'U4']].values, dtype=models[bpm].dtype)
    X[bpm] = torch.tensor(df[bpm]['XPOS'].values, dtype=models[bpm].dtype)
    Y[bpm] = torch.tensor(df[bpm]['YPOS'].values, dtype=models[bpm].dtype)
