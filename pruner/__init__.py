# from . import reg_pruner, oracle_pruner, merge_pruner
# from . import l1_pruner
# from . import iterl1_pruner
# from . import snip_pruner
# from . import lth_pruner
# from . import ssl_pruner, deephoyer_pruner
# from . import taylor_fo_pruner
# from . import mat_pruner
# from . import opp_pruner

# When new pruner implementation is added in the 'pruner' dir, update this dict to maintain minimal code change.
# key: pruning method name, value: the corresponding pruner
# pruner_dict = {
    # 'FixReg': reg_pruner,
    # 'GReg-1': reg_pruner,
    # 'GReg-2': reg_pruner,
    # 'CM': reg_pruner,
    # 'L1': l1_pruner,
    # 'LTH': lth_pruner,
    # 'Oracle': oracle_pruner,
    # 'OPP': opp_pruner,
    # 'Merge': merge_pruner,
    # 'L1_Iter': iterl1_pruner,
    # 'SNIP': snip_pruner,
    # 'SSL': ssl_pruner,
    # 'DeepHoyer': deephoyer_pruner,
    # 'Taylor-FO': taylor_fo_pruner,
    # 'MAT': mat_pruner,
# }

# This is the switch to signal whether pruning is turned on or off
prune_method_arg = 'prune_method'