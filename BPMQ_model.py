import re
import os
import pickle
import time
import datetime
from typing import List, Optional, Dict
from copy import deepcopy as copy

import pandas as pd
import numpy as np
import torch
from torch import nn

from TIS161_coeffs import TIS161_coeffs
try:
    from IPython.display import display as _display
except ImportError:
    _display = print

def display(obj):
    try:
        _display(obj)
    except:
        print(obj)

script_dir = os.path.dirname(os.path.realpath(__file__))


class dummy_scheduler:
    def __init__(self,lr):
        self.lr = lr
    def step(self,*args,**kwargs):
        pass
    def get_last_lr(self,*args,**kwargs):
        return [self.lr]


class BPMQ_model(nn.Module):
    def __init__(self, n_node: int = 16, n_hidden_layer: int = 4, dtype=torch.float32):
        super(BPMQ_model, self).__init__()
        self.pickup_calibration = nn.Parameter(torch.zeros(3, dtype=dtype))
        self.geometry_calibration = nn.Parameter(torch.zeros(1, dtype=dtype))
        
        layers = [nn.Linear(14, n_node), nn.ELU()]
        for _ in range(n_hidden_layer):
            layers.extend([nn.Linear(n_node, n_node), nn.ELU()])
        layers.append(nn.Linear(n_node, 1))
        
        self.nn = nn.Sequential(*layers)
        self.dtype = dtype

    def polynomial_features(self, u: torch.Tensor) -> torch.Tensor:
        poly_features = torch.cat([
            u, 
            u ** 2,
            (u[:, :3] - u[:, 1:]) ** 2, 
            (u[:, :2] - u[:, 2:]) ** 2,
            (u[:, :1] - u[:, 3:]) ** 2,
        ], dim=1)
        return poly_features

    def forward(self, bpm_U: torch.Tensor, bpm_x: torch.Tensor, bpm_y: torch.Tensor) -> torch.Tensor:
        bpm_U = bpm_U.to(dtype=self.dtype)
        c = torch.zeros(4, dtype=self.dtype, device=bpm_U.device)
        c[:3] = self.pickup_calibration
        c[3] = -self.pickup_calibration.sum()

        U = bpm_U / bpm_U.sum(dim=1, keepdim=True)
        u = (1.0 + 0.01 * c.view(1, 4)) * U
    
        Qtheory = (1.0 + 0.1 * self.geometry_calibration) * 241 * ((u[:, 1] + u[:, 2]) - (u[:, 0] + u[:, 3])) \
                  - (bpm_x ** 2 - bpm_y ** 2)
        
        poly_u = self.polynomial_features(4 * u)
        Residual = self.nn(poly_u).view(-1)

        return Qtheory + Residual

        
def BPMQ_loss(model_Q, beam_Q, *args):#, beam_Qerr):
    return torch.mean(torch.abs(model_Q - beam_Q))

       
def train(
    model,
    epochs,lr,
    train_U,train_X,train_Y,train_Q,
    val_U=None,val_X=None,val_Y=None,val_Q=None,
    batch_size=None,
    shuffle=True,
    validation_split=0.0,
    criterion = BPMQ_loss,
    optimizer = torch.optim.Adam,
    optim_args = None,
    optimizer_state_dict = None,
    lr_scheduler = True,
    prev_history = None,
    load_best = True,
    training_timeout = np.inf,
    verbose = False,
    fname_model = 'model.pt',
    fname_opt = 'opt.pt',
    fname_history = 'history.pkl',
    ):
    
    if isinstance(optimizer,str):
        optimizer = getattr(torch.optim, optimizer)
    if isinstance(criterion,str):
        criterion =  getattr(torch.nn, criterion)()

    if verbose:
        print("Train Function Arguments:",datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print(f"  - model: {model.__class__.__name__}")
        print(f"  - x: {x.shape if hasattr(x, 'shape') else type(x)}")
        print(f"  - y: {y.shape if hasattr(y, 'shape') else type(y)}")
        print(f"  - epochs: {epochs}")
        print(f"  - lr: {lr}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - shuffle: {shuffle}")
        print(f"  - validation_split: {validation_split}")
        print(f"  - criterion: {criterion.__class__.__name__}")
        print(f"  - optimizer: {optimizer.__name__}")
        print(f"  - optim_args: {optim_args}")
        print(f"  - optimizer_state_dict: {optimizer_state_dict}")
        print(f"  - lr_scheduler: {lr_scheduler}")
        print(f"  - prev_history: {prev_history}")
        print(f"  - load_best: {load_best}")
        print(f"  - training_timeout: {training_timeout}")
        print(f"  - verbose: {verbose}")
        print(f"  - fname_model: {fname_model}")
        print(f"  - fname_opt: {fname_opt}")
        print(f"  - fname_history: {fname_history}")
        print()
    
    # get dtype and device of the input layer of the model
    n,d = train_U.shape
    if verbose:
        print("Model Paramers:")
    for name, p in model.named_parameters():
        if verbose:
            print(f"  - name: {name}, shape: {p.shape}, dtype: {p.dtype}, device: {p.device}")
        if len(p.shape) == 1:
            continue
        if 'nn.0.' in name:
            device = p.device
            dtype = p.dtype
    if verbose:
        print()

    train_U=torch.tensor(train_U,dtype=dtype)
    train_X=torch.tensor(train_X,dtype=dtype)
    train_Y=torch.tensor(train_Y,dtype=dtype)
    train_Q=torch.tensor(train_Q,dtype=dtype)
    ntrain = len(train_U)
    assert ntrain == len(train_X) == len(train_Y) == len(train_Q)

    nval = 0
    if validation_split>0.0 and val_U is None:
        p = np.random.permutation(np.arange(ntrain))
        train_U = train_U[p]
        train_X = train_X[p]
        train_Y = train_Y[p]
        train_Q = train_Q[p]
    
        nval = int(validation_split*ntrain)
        ntrain = ntrain-nval
        
        val_U = train_U[:nval]
        val_X = train_X[:nval]
        val_Y = train_Y[:nval]
        val_Q = train_Q[:nval]
        train_U = train_U[nval:]
        train_X = train_X[nval:]
        train_Y = train_Y[nval:]
        train_Q = train_Q[nval:]

    elif val_U is not None:
        val_U=torch.tensor(val_U,dtype=dtype)
        val_X=torch.tensor(val_X,dtype=dtype)
        val_Y=torch.tensor(val_Y,dtype=dtype)
        val_Q=torch.tensor(val_Q,dtype=dtype)
        nval = len(val_U)
        assert nval == len(val_X) == len(val_Y) == len(val_Q)
        
    batch_size = batch_size or ntrain
    nbatch_val = int(nval/batch_size)
    if nbatch_val==0 and nval > 0:
        val_batch_size = nval
        nbatch_val = 1
    else:
        val_batch_size = batch_size

    train_batch_size = min(batch_size,ntrain)
    nbatch_train = int(ntrain/train_batch_size)
    
    training_timeout = training_timeout
    t0 = time.monotonic()
    assert epochs>0
    optim_args = optim_args or {}
    

    opt = optimizer(model.parameters(filter(lambda p: p.requires_grad, model.parameters())),lr=lr,**optim_args)
    if optimizer_state_dict is not None:
        opt.load_state_dict(optimizer_state_dict)
        
               
    if prev_history is None:
        history = {
            'train_loss':[],
            'val_loss'  :[],
            'lr'        :[],
            }
    else:
        assert "train_loss" in prev_history
        history = prev_history 
        if "lr" not in history:
            history['lr'] = [None]*len(history["train_loss"])
    epoch_start = len(history['train_loss'])
        
    if lr_scheduler:
        last_epoch = epoch_start*train_batch_size
        if last_epoch == 0:
            last_epoch = -1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
            max_lr=lr,
            div_factor=int(np.clip(epochs/500,a_min=1,a_max=20)),
            pct_start=0.05, 
            final_div_factor=int(np.clip(epochs/50,a_min=10,a_max=1e4)),
            epochs=epochs, steps_per_epoch=nbatch_train, last_epoch=last_epoch)
    else:       
        scheduler = dummy_scheduler(lr)
        
    best = np.inf
    model.train()
    epoch = epoch_start-1
    save_epoch = epoch
    
   
        
    if verbose:
        print("Training begin at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print()
    
    while(True):
        epoch += 1
        if epoch>=epoch_start + epochs:
            break
        lr_ = scheduler.get_last_lr()[0]
        history['lr'].append(lr_)
        
        if shuffle:
            p = np.random.permutation(len(train_U))
            train_U = train_U[p]
            train_X = train_X[p]
            train_Y = train_Y[p]
            train_Q = train_Q[p]
        train_loss = 0
        
        model.train()
        for i in range(nbatch_train):
            i1 = i*train_batch_size
            i2 = i1+train_batch_size
            U = train_U[i1:i2,:].to(device)
            X = train_X[i1:i2].to(device)
            Y = train_Y[i1:i2].to(device)
            Q = train_Q[i1:i2].to(device)
            opt.zero_grad()
            Q_pred = model(U,X,Y)
            loss = criterion(Q_pred, Q)
            loss.backward()
            opt.step()
            scheduler.step()
            train_loss = train_loss + loss.item()
        train_loss /= nbatch_train

        if i2 < ntrain-1 and ntrain < 100:
            U = train_U[i2:,:].to(device)
            X = train_X[i2:].to(device)
            Y = train_Y[i2:].to(device)
            Q = train_Q[i2:].to(device)
            opt.zero_grad()
            Q_pred = model(U,X,Y)
            loss = criterion(Q_pred, Q)
            loss.backward()
            opt.step()
            train_loss = (train_loss*train_batch_size*nbatch_train + loss.item()*(ntrain-i2))/ntrain
        history['train_loss'].append(train_loss)

        val_loss = 0.0
        if nbatch_val>0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for i in range(nbatch_val):
                    i1 = i*val_batch_size
                    i2 = i1+val_batch_size
                    U = val_U[i1:i2,:].to(device)
                    X = val_X[i1:i2].to(device)
                    Y = val_Y[i1:i2].to(device)
                    Q = val_Q[i1:i2].to(device)
                    Q_pred = model(U,X,Y)
                    loss = criterion(Q, Q_pred)
                    val_loss += loss.item()
                val_loss /= nbatch_val

                if i2 < nval-1:
                    U = val_U[i2:,:].to(device)
                    X = val_X[i2:].to(device)
                    Y = val_Y[i2:].to(device)
                    Q = val_Q[i2:].to(device)
                    Q_pred = model(U,X,Y)
                    loss = criterion(Q, Q_pred)
                    val_loss = (val_loss*val_batch_size*nbatch_val + loss.item()*(nval-i2))/nval
            history['val_loss'].append(val_loss)

            if val_loss < best:
                best = val_loss
                model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(model_state_dict,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        else:
            if train_loss < best:
                best = train_loss
                model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(model_state_dict,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        
        if verbose:
            nskip = int(epochs/100)
            if epoch%nskip==0:
                elapsed_t = datetime.timedelta(seconds=time.monotonic() - t0)
                if nbatch_val>0:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | Val Loss: {val_loss:.2E} | lr: {lr_:.2E} | {elapsed_t}')
                else:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | lr: {lr_:.2E} | {elapsed_t}')

    dt = time.monotonic()-t0                
    if load_best:
        model.load_state_dict(model_state_dict)
            
    return history,model_state_dict,opt_state_dict

        
def sort_by_Dnum(strings):
    """
    Sort a list of PVs by dnum.
    """
    # Define a regular expression pattern to extract the 4-digit number at the end of each string
    pattern = re.compile(r'\D(\d{4})$')

    # Define a custom sorting key function that extracts the 4-digit number using the regex pattern
    def sorting_key(s):
        match = pattern.search(s)
        if match:
            return int(match.group(1))
        return 0  # Default value if no match is found

    # Sort the strings based on the custom sorting key
    sorted_strings = sorted(strings, key=sorting_key)
    return sorted_strings


class raw2Q_processor:
    def __init__(self,
        BPM_names  : List[str],
        BPMQ_models: Optional[Dict[str,torch.nn.Module]] = None):
        
        self.BPM_names = sort_by_Dnum(BPM_names)     
        BPM_TIS161_PVs = []   
        calibrated_PVs = []
        BPM_TIS161_coeffs = np.zeros(4*len(self.BPM_names))
        PVs2read = []

        if BPMQ_models is None:
            self.BPMQ_models = {name:None for name in self.BPM_names}
        else:
            self.BPMQ_models = BPMQ_models

        for i,name in enumerate(self.BPM_names):
            if name not in TIS161_coeffs:
                raise ValueError(f"{name} not found in TIS161_coeffs")
            TIS161_PVs = [f"{name}:TISMAG161_{i + 1}_RD" for i in range(4)]
            calibrated_PVs += [f"{name}:U{i + 1}" for i in range(4)]
            BPM_TIS161_PVs += TIS161_PVs
            BPM_TIS161_coeffs[4*i:4*(i+1)] = TIS161_coeffs[name]
            PVs2read += BPM_TIS161_PVs + [
                f"{name}:{tag}" for tag in ["XPOS_RD", "YPOS_RD", "PHASE_RD", "MAG_RD", "CURRENT_RD"
                ]]
            if self.BPMQ_models[name] is None:
                try:
                    # load default BPMQ model
                    fname = name.replace('_D','')[-7:]
                    state_dict = torch.load(os.path.join(script_dir,fname,'model.pt'))
                    model_info = {'n_node':len(state_dict['nn.2.bias']),
                                  'n_hidden_layer':int((len(state_dict)-2)/2)-2,
                                  'dtype':state_dict['nn.2.bias'].dtype}
                    model = BPMQ_model(**model_info)
                    model.load_state_dict(state_dict)
                    self.BPMQ_models[name] = model
                except Exception as e:
                    print(f"Failed to load BPMQ model for {name}: {e}. BPMQ formula will be used instead.")
                
        self.PVs2read = PVs2read
        self.BPM_TIS161_PVs = BPM_TIS161_PVs
        self.calibrated_PVs = calibrated_PVs
        self.BPM_TIS161_coeffs = np.array(BPM_TIS161_coeffs)
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        df: DataFrame whose columns should include BPM_name:TIS161_i_RD, XPOS, YPOS,
        '''
        df[self.calibrated_PVs] = df[self.BPM_TIS161_PVs].values*self.BPM_TIS161_coeffs[None,:]        
    
        for i,name in enumerate(self.BPM_names):
            U = self.calibrated_PVs[4*i:4*(i+1)]
            model = self.BPMQ_models[name]
            if model:
                with torch.no_grad():
                    u_ = torch.tensor(df[U].values,dtype=model.dtype)
                    x_ = torch.tensor(df[name+':XPOS_RD'].values,dtype=model.dtype)
                    y_ = torch.tensor(df[name+':YPOS_RD'].values,dtype=model.dtype)
                    #print("====",name,"====")
                    #print("u_:",u_)
                    #print(",model(u_,x_,y_).cpu().numpy()",,model(u_,x_,y_).cpu().numpy())
                    df[name+':beamQ'] = model(u_,x_,y_).cpu().numpy()
            else:
                diffsum = (df[[U[1],U[2]]].sum(axis=1) -df[[U[0],U[3]]].sum(axis=1)) / df[U].sum(axis=1)
                df[name+':beamQ'] = (241*diffsum - (df[name+':XPOS_RD']**2 - df[name+':YPOS_RD']**2))
        return df
