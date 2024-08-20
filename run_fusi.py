from random import randint

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import torch.nn as nn
from torch.utils.data import dataset

from models import EMModule, RecurrentContextModule
import utils
from datasets import *
    

def gen_data_loader(paradigm):
    if paradigm=='blocked':
        contexts_to_load = [0,1,0,1] + [randint(0,1) for _ in range(40)]
        n_samples_per_context = [40,40,40,40] + [1]*40
        ds = FusiDataset(n_samples_per_context, contexts_to_load)
    elif paradigm == 'interleaved':
        contexts_to_load = [0,1]*80 + [randint(0,1) for _ in range(40)]
        n_samples_per_context = [1]*160 + [1]*40
        ds = FusiDataset(n_samples_per_context, contexts_to_load)
    else:
        raise Exception('Illegal dataset paradigm')
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    

''' Model preparation functions '''
def prep_recurrent_network(rnet, state_d, persistance=-0.6):
    with torch.no_grad():
        rnet.state_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.state_to_hidden.bias.zero_()
        rnet.hidden_to_hidden.weight.zero_()
        rnet.hidden_to_hidden.bias.zero_()
        rnet.state_to_hidden_wt.weight.zero_()
        rnet.state_to_hidden_wt.bias.copy_(torch.ones((len(rnet.state_to_hidden_wt.bias),), dtype=torch.float) * persistance)
        rnet.hidden_to_hidden_wt.weight.zero_()
        rnet.hidden_to_hidden_wt.bias.zero_()
        # Set hidden to context weights as an identity matrix.
        rnet.hidden_to_context.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.hidden_to_context.bias.zero_()

    # Set requires_grad to True for hidden_to_context.weight before freezing other parameters
    rnet.hidden_to_context.weight.requires_grad = True
    rnet.hidden_to_context.bias.requires_grad = True

    # Freeze recurrent weights to stabilize training
    for name, p in rnet.named_parameters():
        if 'hidden_to_context' not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    return rnet


def gen_model(params):
    context_module = RecurrentContextModule(params.output_d, params.output_d, params.output_d)
    em_module = EMModule(params.temperature)
    context_module = prep_recurrent_network(context_module, params.output_d, params.persistance)
    return context_module, em_module

def calc_prob(em_preds, test_ys):
    # only consider the terminal three states (they are the only predictable transitions).
    em_preds_new, test_ys_new = em_preds[:, 2:-1, :], test_ys[:, 2:-1, :]
    em_probability = (em_preds_new*test_ys_new).sum(-1).mean(-1)
    trial_probs = (em_preds*test_ys) 
    return em_probability, trial_probs

def run_participant(params, training_paradigm):
    performance_data = {'seed':[], 'paradigm':[], 'trial':[], 'probability':[]}
    loss_fn = nn.BCELoss()
    data_loader = gen_data_loader(training_paradigm)
    context_module, em_module = gen_model(params)
    optimizer = torch.optim.SGD([{'params': context_module.parameters(), 'lr': params.episodic_lr}])
    em_preds = []
    em_contexts = []
    em_probs = []
    err_vec = torch.ones([1,2])
    for trial, (x,_,y) in enumerate(data_loader):
        for _ in range(params['n_optimization_steps']):
            context = context_module(err_vec)
            if trial > 0:
                optimizer.zero_grad()
                pred_em = em_module(x,context)
                loss = loss_fn(pred_em, y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    err_vec = (pred_em-y)**2
            else:
                pred_em = torch.zeros([1,params.output_d]).float()
        with torch.no_grad():
            em_module.write(x,context,y)
            em_preds.append(pred_em.cpu().detach().numpy())
            em_contexts.append(context.cpu().detach().numpy())

    # Collect some training data for analysis.
    em_preds = np.stack(em_preds).squeeze().reshape(-1,4,params.output_d)
    test_ys = data_loader.dataset.ys.cpu().numpy().reshape(-1,4,params.output_d)
    correct_prob, trial_probs = calc_prob(em_preds, test_ys)
    em_probs.append(trial_probs)
    performance_data['probability'].extend(correct_prob)
    performance_data['seed'].extend([params.seed]*len(correct_prob))
    performance_data['paradigm'].extend([training_paradigm]*len(correct_prob))
    performance_data['trial'].extend(list(range(len(correct_prob))))
    return pd.DataFrame(performance_data), em_probs, em_contexts 


def run_experiment(params):
    performance_data = []
    correct_probs = []
    context_reps = []
    for i in range(params.n_participants):
        utils.set_random_seed(i)
        for training_paradigm in params['paradigms']:
            participant_df, em_probs, em_contexts = run_participant(params, training_paradigm)
            performance_data.append(participant_df)
            correct_probs.append(em_probs)
            context_reps.append(em_contexts)
    exp_df = pd.concat(performance_data).reset_index(drop=True)
    correct_probs = np.stack(correct_probs)
    context_reps = np.stack(context_reps)
    return exp_df, correct_probs, context_reps