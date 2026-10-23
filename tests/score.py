"""Fixed positive behavior weights; scientific failures reduce reward monotonically."""
from collections import Counter


def aggregate(nodeids,reports,profile):
    functions=[node.split('::')[-1].split('[')[0] for node in nodeids]
    counts=Counter(functions)
    weights=profile['weights']
    if set(counts)!=set(weights) or sum(weights.values())!=profile['total_weight']:
        raise ValueError('Scoring profile does not match the complete suite')
    earned=0.0;breakdown=[]
    for node,function in zip(nodeids,functions):
        weight=weights[function]/counts[function]
        passed=reports[node]['call']['outcome']=='passed'
        contribution=weight if passed else 0.0
        earned+=contribution
        breakdown.append({'nodeid':node,'weight':weight,'passed':passed,'earned':contribution})
    return earned/profile['total_weight'],breakdown
