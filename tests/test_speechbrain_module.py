def test_speechbrain_module():
    from speechbrain.module import SpeechBrainModule

    class TestClass(SpeechBrainModule):
        def __init__(option1 = 1, **kwargs):
            options = {'option1': 'int(1, inf)', 'value': option1}
            expected_inputs = [{'type': 'torch.Tensor', 'shape': [2, 3]}]
            def hook(self, input):
                input = input.transpose()
            super().__init__(options, expected_inputs, hook, **kwargs)

    sbmodule = TestClass()
    assert sbmodule.option1 == 1
    sbmodule = TestClass(2)
    assert sbmodule.option1 == 2
