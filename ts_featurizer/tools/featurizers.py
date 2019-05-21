"""
All these Featurizers have not been implemented, but are put here as an example of how the new Featurizers
can have a different functionality, overriding the BaseFeaturizer's functions.

"""
from .base import BaseFeaturizer


class TimeFeaturizer(BaseFeaturizer):

	def __init__(self, config=None, ):
		super(TimeFeaturizer, self).__init__(config, )


class FrequencyFeaturizer(BaseFeaturizer):

	def __init__(self, config=None, ):
		super(FrequencyFeaturizer, self).__init__(config, )


class AutoRegressionFeaturizer(BaseFeaturizer):

	def __init__(self, config=None, ):
		super(AutoRegressionFeaturizer, self).__init__(config, )


class HilbertFeaturizer(BaseFeaturizer):

	def __init__(self, config=None, ):
		super(HilbertFeaturizer, self).__init__(config, )


class WaveletFeaturizer(BaseFeaturizer):

	def __init__(self, config=None, ):
		super(WaveletFeaturizer, self).__init__(config, )
