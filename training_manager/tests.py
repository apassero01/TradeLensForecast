from copy import deepcopy

import numpy as np
import pandas as pd
from django.test import TestCase
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

from sequenceset_manager.models import SequenceSet
from sequenceset_manager.services import SequenceSetService
from training_manager.models import Evaluation, Trainer, TrainingSession, FeatureSet













