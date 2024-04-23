import pandas as pd
from scipy.stats import entropy

## Self-implemented function: information gain
def information_gain(essentiality, cluster_exp):
    entropy_before = entropy(essentiality.value_counts(normalize=True),base=2) #ok
    grouped_value = essentiality.groupby(cluster_exp).\
                    value_counts(normalize=True).\
                    reset_index(name='count')
    high_exp = grouped_value[grouped_value.Disc_Exp == 'High']['count']
    low_exp = grouped_value[grouped_value.Disc_Exp == 'Low']['count']
    high_exp_entropy = entropy(high_exp,base=2)
    low_exp_entropy = entropy(low_exp,base=2)

    exp_prop = cluster_exp.value_counts(normalize=True)
    entropy_after = exp_prop['Low']*low_exp_entropy + exp_prop['High']*high_exp_entropy
    gain = entropy_before - entropy_after
    return (gain)
