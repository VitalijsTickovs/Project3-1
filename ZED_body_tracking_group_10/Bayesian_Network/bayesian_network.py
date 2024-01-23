import pickle

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from pathlib import Path

import os

print(os.getcwd())


with open('/home/kamil/PycharmProjects/Project3-1_WORKING_ZED/ZED_body_tracking_group_10/Bayesian_Network/models/recent.pkl', 'rb') as file:
    model = pickle.load(file)
    assert model.check_model()
    inference = VariableElimination(model)


def get_weights(evidence:dict):
    print(evidence)
    result = inference.query(
        variables=['NextObject'],
        evidence=evidence
    )
    values = result.values[result.state_names[result.variables[0]]]
    weights = {
        'Crate': values[0],
        'Feeder': values[1],
        'Cup': values[2]
        
    }
    return weights


if __name__ == '__main__':
    
    evidence={
        'CrateAvailable': 1,
        'FeederAvailable': 1,
        'CupAvailable': 1,

        'Worker': 0
    }

    weights = get_weights(evidence)
    print(weights)