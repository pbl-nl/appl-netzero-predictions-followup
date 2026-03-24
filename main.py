from stages.download import run as download_data
from stages.process import run as process_data
from stages.merge_target import run as merge_data

download_data()
process_data()
merge_data()
