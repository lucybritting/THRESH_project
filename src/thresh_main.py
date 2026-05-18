from utils.range_merge import WidestRange, AverageRange, NarrowestRange, MergeWarningLog


RANGE_MERGE_STRATEGIES = [WidestRange(), AverageRange(warning_log=MergeWarningLog()), NarrowestRange(warning_log=MergeWarningLog())]

# TODO: iterate over the merge strategies to apply them

# save the warnings of the range merge strategies
for strategy in RANGE_MERGE_STRATEGIES:
    # TODO: set some output dir for saving the warning csv files
    output = None
    if strategy.warning_log is not None:
        strategy.warning_log.save(output)
