from utils.range_merge import WidestRange, AverageRange, NarrowestRange, MergeWarningLog
from utils.representations import BinaryRepresentation, ContinuousRepresentation, DiscreteRepresentation, DiscreteBinaryRepresentation, AllRepresentation

RANGE_MERGE_STRATEGIES = [WidestRange(), AverageRange(warning_log=MergeWarningLog()), NarrowestRange(warning_log=MergeWarningLog())]
ENCODING_STRATEGIES = [BinaryRepresentation(), ContinuousRepresentation(), DiscreteRepresentation(), DiscreteBinaryRepresentation(), AllRepresentation()]




# save the warnings of the range merge strategies
# TODO: only iterate through the range merge strategies if want to test the different strategies. Per default use the WidestRange strategy
for strategy in RANGE_MERGE_STRATEGIES:
    # TODO: set some output dir for saving the warning csv files
    output = None
    if strategy.warning_log is not None:
        strategy.warning_log.save(output)