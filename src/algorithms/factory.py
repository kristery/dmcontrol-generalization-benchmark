from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.sac_exp import SAC_EXP
from algorithms.sac_feat import SAC_FEAT
from algorithms.sac_feat_exp import SAC_FEAT_EXP

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
        'sac_exp': SAC_EXP,
        'sac_feat': SAC_FEAT,
        'sac_feat_exp': SAC_FEAT_EXP
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
