from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


probability_categories = {
    'Low': (0.0, 0.33),
    'Medium': (0.34, 0.66),
    'High': (0.67, 1.0)
}

def probability_to_category(probability, categories):
    for category, (lower_bound, upper_bound) in categories.items():
        if lower_bound <= probability <= upper_bound:
            return category 


model = BayesianNetwork([
    ('crate_present', 'human_action'),
    ('feeder_present', 'human_action'),
    ('cup_present', 'human_action'),
    ('skeleton_status', 'human_action')
])





