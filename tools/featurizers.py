from tools.base import BaseFeaturizer


class TimeFeaturizer(BaseFeaturizer):

	def __init__(self, data, config=None, ):
		super(TimeFeaturizer, self).__init__(data, config, )


class FrequencyFeaturizer(BaseFeaturizer):

	def __init__(self, data, config=None,):
		super(FrequencyFeaturizer, self).__init__(data, config, )


class AutoRegressionFeaturizer(BaseFeaturizer):

	def __init__(self, data, config=None, ):
		super(AutoRegressionFeaturizer, self).__init__(data, config, )


class WaveletFeaturizer(BaseFeaturizer):

	def __init__(self, data, config=None, ):
		super(WaveletFeaturizer, self).__init__(data, config, )
