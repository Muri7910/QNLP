import torch
from torch.nn import CrossEntropyLoss
from lambeq import BobcatParser
import numpy as np

epochs = 100

train_data = ['cat runs on land']
train_labels = [[0., 0., 0., 1.,0., 0., 0., 0.]]

parser = BobcatParser()

train_diagrams = parser.sentences2diagrams(train_data)

loss = CrossEntropyLoss()

def CEL(y_hat, y):
    flattened_y_hat = y_hat.flatten()
    return float(loss(torch.tensor(flattened_y_hat), torch.tensor(y[0])))


def acc(y_hat, y):
    a=y_hat.flatten()
    #b=y_hat
    max_index = np.argmax(a)
    new_arr = np.zeros(a.shape)
    new_arr[max_index] = 1
    if np.array_equal(new_arr, y[0]):
        return 1
    else:
        return 0

from lambeq import AtomicType, IQPAnsatz

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE
ansatz = IQPAnsatz(ob_map={N: 1, S: 3, P: 1}, n_layers=1, n_single_qubit_params=1)
train_circuits = [ansatz(diagram) for diagram in train_diagrams]


from lambeq import NumpyModel
model = NumpyModel.from_diagrams(train_circuits)
from lambeq import QuantumTrainer, SPSAOptimizer

# here you can play around with hyperparameters
trainer = QuantumTrainer(
    model,
    loss_function=CEL,
    epochs=epochs,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': 0.2, 'c': 0.06, 'A':0.01*epochs},
    evaluate_functions={'acc': acc},
    evaluate_on_train=True,
    verbose = 'text',
    seed=0
)

from lambeq import Dataset

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=1,
            shuffle=False)

trainer.fit(train_dataset, logging_step=1)