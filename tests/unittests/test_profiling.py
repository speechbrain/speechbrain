
def test_profile_class(device):
    import torch
    from torch.optim import SGD
    from speechbrain.core import Brain
    from speechbrain.utils.profiling import profile

    @profile
    class SimpleBrain(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    """
    This way of using ``@profile`` has a very brief summary, since the constructor is benchmarked.
    
    Example:
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
        aten::_has_compatible_shallow_copy_type        99.45%     907.000us        99.45%     907.000us     226.750us             4  
                                       aten::to         0.55%       5.000us         0.55%       5.000us       2.500us             2  
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 912.000us
    
    The later call `brain.fit` is not re-calling the profiler.
    """

    model = torch.nn.Linear(in_features=10, out_features=10, device=device)
    inputs = torch.rand(10, 10, device=device)
    targets = torch.rand(10, 10, device=device)
    train_set = ([inputs, targets],)
    valid_set = ([inputs, targets],)

    brain = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )

    assert brain.profiler is not None
    # TODO asserts
    print(brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # TODO manual work around is necessary (start & stop)
    brain.profiler.start()
    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    brain.profiler.stop()
    # TODO asserts
    print(brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    """
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                              aten::l1_loss         2.44%     311.000us        25.95%       3.312ms      82.800us            40  
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        12.47%       1.591ms        19.39%       2.475ms      61.875us            40  
                                               aten::linear         1.42%     181.000us        13.16%       1.680ms      84.000us            20  
                                                   aten::to         2.12%     271.000us        10.70%       1.366ms       8.537us           160  
                                             aten::isfinite         1.95%     249.000us        10.01%       1.278ms      42.600us            30  
                                                 aten::mean         1.21%     155.000us         9.38%       1.197ms      59.850us            20  
                                             aten::_to_copy         5.59%     713.000us         8.96%       1.143ms      11.430us           100  
                                                aten::stack         2.19%     279.000us         8.39%       1.071ms      21.420us            50  
                                               aten::matmul         1.86%     238.000us         7.94%       1.013ms      50.650us            20  
                                     aten::l1_loss_backward         1.36%     174.000us         7.15%     912.000us      45.600us            20  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------      
    """


def test_profile_func(device):
    import torch
    from torch.optim import SGD
    from speechbrain.core import Brain
    from speechbrain.utils.profiling import profile, events_diff
    from torch.autograd.profiler import record_function

    class SimpleBrain(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    class SimpleBrainNittyGritty(Brain):
        def compute_forward(self, batch, stage):
            # example: one way of using torch.autograd.profiler.record_function
            with record_function("is this faster (?)"):
                this = self.modules.model(batch[0])
            return this

        def compute_objectives(self, predictions, batch, stage):
            # example: one could also think of running comparative testing using record_function
            with record_function("or that (?)"):
                that = torch.nn.functional.l1_loss(predictions, batch[1])
            return that
        """
        Both classes have alike numbers of function calls (given the same input data and train function).
        Sparing where both have the same number of calls:
        
        SimpleBrain
        ----------------  ------------  ------------  ------------  ------------  ------------  ------------  
                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
        ----------------  ------------  ------------  ------------  ------------  ------------  ------------  
             aten::zeros        17.44%     240.000us        27.69%     381.000us       6.350us            60  
             aten::empty        26.96%     371.000us        27.11%     373.000us       1.492us           250  
            aten::detach        18.68%     257.000us        25.65%     353.000us       5.694us            62  
              aten::add_        25.00%     344.000us        25.00%     344.000us       5.931us            58  
                  detach         6.98%      96.000us         9.45%     130.000us       2.097us            62  
             aten::zero_         4.94%      68.000us         4.94%      68.000us       0.756us            90  
        ----------------  ------------  ------------  ------------  ------------  ------------  ------------  
        Self CPU time total: 1.376ms

        SimpleBrainNittyGritty
        ----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                          Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
        ----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
            is this faster (?)        29.70%       1.024ms        76.19%       2.627ms     131.350us            20  
                   or that (?)        29.29%       1.010ms        67.00%       2.310ms     115.500us            20  
                   aten::zeros         8.32%     287.000us        15.52%     535.000us       5.350us           100  
                   aten::empty        14.01%     483.000us        14.07%     485.000us       1.470us           330  
                    aten::add_         9.92%     342.000us         9.92%     342.000us       5.700us            60  
                  aten::detach         2.81%      97.000us         6.09%     210.000us       3.500us            60  
                        detach         3.28%     113.000us         4.00%     138.000us       2.300us            60  
                   aten::zero_         2.67%      92.000us         2.67%      92.000us       0.708us           130  
        ----------------------  ------------  ------------  ------------  ------------  ------------  ------------ 
        Self CPU time total: 3.448ms
        
        Curiosity doesn't come entirely for free ;-)
        """

    @profile
    def train(brain, train_set, valid_set):
        brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)

    model = torch.nn.Linear(in_features=10, out_features=10, device=device)
    inputs = torch.rand(10, 10, device=device)
    targets = torch.rand(10, 10, device=device)
    training_set = ([inputs, targets],)
    validation_set = ([inputs, targets],)

    simple_brain = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )

    prof_simple = train(simple_brain, training_set, validation_set)
    print(prof_simple.key_averages().table(sort_by="cpu_time_total"))
    # TODO asserts

    simple_brain_nitty_gritty = SimpleBrainNittyGritty(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    prof_nitty_gritty = train(simple_brain_nitty_gritty, training_set, validation_set)
    print(prof_nitty_gritty.key_averages().table(sort_by="cpu_time_total"))

    events_diff(prof_simple.key_averages(), prof_nitty_gritty.key_averages())
