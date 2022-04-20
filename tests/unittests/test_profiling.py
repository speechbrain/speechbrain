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

    model = torch.nn.Linear(in_features=10, out_features=10, device=device)
    inputs = torch.rand(10, 10, device=device)
    targets = torch.rand(10, 10, device=device)
    train_set = ([inputs, targets],)
    valid_set = ([inputs, targets],)

    # Profiling: __init__ constructor.
    brain = SimpleBrain({"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device})
    assert brain.profiler is not None
    assert brain.profiler.profiler is not None
    assert len(brain.profiler.key_averages()) == 2
    assert brain.profiler.events().total_average().count == 6
    assert len(brain.profiler.speechbrain_parsed_kineto_traces) == 1  # set & filled by the @profile decorator
    """print(brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
        aten::_has_compatible_shallow_copy_type        76.92%      10.000us        76.92%      10.000us       2.500us             4  
                                       aten::to        23.08%       3.000us        23.08%       3.000us       1.500us             2  
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 13.000us
    """

    # Profiling: fit() for train operations.
    # By default, @profile should also annotate fit & evaluate functions; here the fit function is tested only.
    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    assert brain.profiler is not None
    assert len(brain.profiler.key_averages()) == 72
    assert brain.profiler.events().total_average().count == 2832
    assert len(brain.profiler.speechbrain_parsed_kineto_traces) == 2
    assert len(brain.profiler.speechbrain_parsed_kineto_traces[0]) == 6
    assert len(brain.profiler.speechbrain_parsed_kineto_traces[1]) == 2862
    """print(brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                              aten::l1_loss         2.33%     429.000us        27.51%       5.076ms     126.900us            40  
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        12.06%       2.225ms        19.05%       3.515ms      87.875us            40  
                                               aten::linear         1.19%     220.000us        12.88%       2.376ms     118.800us            20  
                                                   aten::to         1.76%     325.000us        11.86%       2.188ms      13.675us           160  
                                                 aten::mean         1.12%     206.000us        10.43%       1.924ms      96.200us            20  
                                             aten::_to_copy         6.95%       1.282ms        10.19%       1.881ms      18.810us           100  
                                             aten::isfinite         1.95%     359.000us         9.79%       1.807ms      60.233us            30  
                                                aten::stack         2.14%     395.000us         8.63%       1.592ms      31.840us            50  
                                                 aten::div_         1.55%     286.000us         7.78%       1.435ms      71.750us            20  
                                               aten::matmul         1.58%     292.000us         7.57%       1.397ms      69.850us            20  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 18.451ms
    """

    # Profiling: putting previous benchmark reporting together.
    full_report = brain.profiler.merge_traces()
    assert len(full_report) == 2838
    assert len(full_report.key_averages()) == 73
    """print(full_report.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                              aten::l1_loss         2.32%     429.000us        27.49%       5.076ms     126.900us            40  
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        12.05%       2.225ms        19.04%       3.515ms      87.875us            40  
                                               aten::linear         1.19%     220.000us        12.87%       2.376ms     118.800us            20  
                                                   aten::to         1.78%     328.000us        11.87%       2.191ms      13.525us           162  
                                                 aten::mean         1.12%     206.000us        10.42%       1.924ms      96.200us            20  
                                             aten::_to_copy         6.94%       1.282ms        10.19%       1.881ms      18.810us           100  
                                             aten::isfinite         1.94%     359.000us         9.79%       1.807ms      60.233us            30  
                                                aten::stack         2.14%     395.000us         8.62%       1.592ms      31.840us            50  
                                                 aten::div_         1.55%     286.000us         7.77%       1.435ms      71.750us            20  
                                               aten::matmul         1.58%     292.000us         7.57%       1.397ms      69.850us            20  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 18.464ms
    """


def test_profile_func(device):
    import torch
    from pytest import raises
    from torch.optim import SGD
    from speechbrain.core import Brain
    from torch.autograd.profiler import record_function
    from speechbrain.utils.profiling import profile, events_diff

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
    # print(prof_simple.key_averages().table(sort_by="cpu_time_total"))
    assert len(prof_simple.events()) == 2832
    assert len(prof_simple.key_averages()) == 72

    simple_brain_nitty_gritty = SimpleBrainNittyGritty(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    prof_nitty_gritty = train(simple_brain_nitty_gritty, training_set, validation_set)
    # print(prof_nitty_gritty.key_averages().table(sort_by="cpu_time_total"))
    assert len(prof_nitty_gritty.events()) == 3030
    assert len(prof_nitty_gritty.key_averages()) == 74

    # The outputs of this diff are only for visualisation, ``simple_delta._build_tree()`` will throw an error.
    simple_delta, nitty_gritty_delta = events_diff(prof_simple.key_averages(), prof_nitty_gritty.key_averages())
    assert len(simple_delta) == 6
    assert len(nitty_gritty_delta) == 8
    assert simple_delta.total_average().count == 582
    assert nitty_gritty_delta.total_average().count == 780
    with raises(Exception) as err_tree:
        simple_delta._build_tree()  # as mentioned.
    assert err_tree.type == AttributeError
    with raises(Exception) as err_averages:
        simple_delta.key_averages()  # as mentioned.
    assert err_averages.type == AssertionError
    """Both classes have alike numbers of function calls (given the same input data and train function).
    Sparing where both have the same number of calls:
    
    print(simple_delta.table(sort_by="cpu_time_total"))
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

    print(nitty_gritty_delta.table(sort_by="cpu_time_total"))
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
    
    Curiosity doesn't come for free ;-)
    """


def test_scheduler(device):
    import torch
    from pytest import raises
    from torch.optim import SGD
    from speechbrain.core import Brain
    from speechbrain.utils.profiling import profile, scheduler

    @scheduler
    @profile
    class SimpleBrain(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    model = torch.nn.Linear(in_features=10, out_features=10, device=device)
    inputs = torch.rand(10, 10, device=device)
    targets = torch.rand(10, 10, device=device)
    train_set = ([inputs, targets],)
    valid_set = ([inputs, targets],)
    test_set = ([inputs, targets],)

    # Profiling: __init__ constructor -- while scheduler: waiting --> nothing to report
    brain = SimpleBrain({"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device})
    assert brain.profiler.profiler is None
    assert len(brain.profiler.speechbrain_parsed_kineto_traces) == 0
    with raises(Exception) as err:
        brain.profiler.events()  # Tracing hasn't started, yet, so everything is in err. Scheduler says: wait.
    assert err.type == AssertionError
    assert brain.profiler.step_num == 0

    # Profiling: fit() for train operations.
    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    assert brain.profiler.step_num == 20
    assert len(brain.profiler.speechbrain_parsed_kineto_traces) == 1
    assert len(brain.profiler.events()) == 293
    assert len(brain.profiler.key_averages()) == 73
    """print(brain.profiler.key_averages().table(sort_by="cpu_time_total"))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                              ProfilerStep*        55.48%       1.504ms        99.00%       2.684ms       1.342ms             2  
                                              aten::l1_loss         1.07%      29.000us         9.30%     252.000us      63.000us             4  
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...         4.57%     124.000us         7.01%     190.000us      47.500us             4  
                                               aten::linear         2.32%      63.000us         6.93%     188.000us      94.000us             2  
                                             aten::isfinite         0.89%      24.000us         5.35%     145.000us      48.333us             3  
                                                   aten::to         1.00%      27.000us         4.46%     121.000us       7.562us            16  
                                             aten::_to_copy         2.18%      59.000us         3.58%      97.000us       9.700us            10  
                                                aten::stack         0.89%      24.000us         3.28%      89.000us      17.800us             5  
                                     aten::l1_loss_backward         0.55%      15.000us         3.14%      85.000us      42.500us             2  
                                                 aten::mean         0.48%      13.000us         3.10%      84.000us      42.000us             2  
                                               aten::matmul         0.55%      15.000us         2.84%      77.000us      38.500us             2  
                                                   aten::ne         0.77%      21.000us         2.73%      74.000us      24.667us             3  
                                                   aten::mm         2.29%      62.000us         2.36%      64.000us      21.333us             3  
                                                  aten::div         1.36%      37.000us         2.25%      61.000us      20.333us             3  
                                    Optimizer.step#SGD.step         1.88%      51.000us         2.21%      60.000us      60.000us             1  
    autograd::engine::evaluate_function: L1LossBackward0...         0.11%       3.000us         2.07%      56.000us      56.000us             1  
                                                aten::zeros         1.44%      39.000us         1.99%      54.000us       6.750us             8  
                                            L1LossBackward0         0.11%       3.000us         1.95%      53.000us      53.000us             1  
                                                    aten::t         0.89%      24.000us         1.88%      51.000us      10.200us             5  
                                                  aten::cat         0.59%      16.000us         1.81%      49.000us       9.800us             5  
                                                 aten::div_         0.70%      19.000us         1.73%      47.000us      23.500us             2  
           autograd::engine::evaluate_function: MmBackward0         0.11%       3.000us         1.70%      46.000us      46.000us             1  
                                                  aten::abs         1.14%      31.000us         1.59%      43.000us       5.375us             8  
                                                MmBackward0         0.37%      10.000us         1.59%      43.000us      43.000us             1  
                                                  aten::sum         1.00%      27.000us         1.33%      36.000us      12.000us             3  
                                                aten::empty         1.29%      35.000us         1.29%      35.000us       1.207us            29  
          autograd::engine::evaluate_function: AddBackward0         0.63%      17.000us         1.25%      34.000us      34.000us             1  
                                                 aten::_cat         0.85%      23.000us         1.22%      33.000us       6.600us             5  
                                                 aten::add_         1.11%      30.000us         1.11%      30.000us       5.000us             6  
                                            aten::transpose         0.70%      19.000us         0.92%      25.000us       5.000us             5  
          autograd::engine::evaluate_function: DivBackward0         0.18%       5.000us         0.92%      25.000us      25.000us             1  
                                                 aten::norm         0.77%      21.000us         0.89%      24.000us       8.000us             3  
                          Optimizer.zero_grad#SGD.zero_grad         0.63%      17.000us         0.81%      22.000us      22.000us             1  
                                                 aten::item         0.55%      15.000us         0.77%      21.000us       3.000us             7  
                                                aten::copy_         0.77%      21.000us         0.77%      21.000us       2.100us            10  
                                               DivBackward0         0.15%       4.000us         0.74%      20.000us      20.000us             1  
    autograd::engine::evaluate_function: torch::autograd...         0.15%       4.000us         0.74%      20.000us      10.000us             2  
                                                  aten::mul         0.55%      15.000us         0.74%      20.000us       5.000us             4  
                                               aten::detach         0.37%      10.000us         0.70%      19.000us       3.167us             6  
                                           aten::as_strided         0.59%      16.000us         0.63%      17.000us       1.062us            16  
                                            aten::unsqueeze         0.41%      11.000us         0.59%      16.000us       2.667us             6  
                                        aten::empty_strided         0.59%      16.000us         0.59%      16.000us       1.455us            11  
                            torch::autograd::AccumulateGrad         0.30%       8.000us         0.59%      16.000us       8.000us             2  
                                           aten::is_nonzero         0.18%       5.000us         0.59%      16.000us       5.333us             3  
                                                 aten::view         0.55%      15.000us         0.55%      15.000us       3.000us             5  
                                              aten::random_         0.52%      14.000us         0.52%      14.000us       7.000us             2  
    autograd::engine::evaluate_function: UnsafeViewBackw...         0.11%       3.000us         0.52%      14.000us      14.000us             1  
                                                  aten::sub         0.48%      13.000us         0.48%      13.000us       4.333us             3  
                                                  aten::add         0.18%       5.000us         0.48%      13.000us      13.000us             1  
                                            aten::ones_like         0.15%       4.000us         0.44%      12.000us      12.000us             1  
                                                 aten::mul_         0.44%      12.000us         0.44%      12.000us       4.000us             3  
                                                     detach         0.33%       9.000us         0.44%      12.000us       2.000us             6  
                                        UnsafeViewBackward0         0.11%       3.000us         0.41%      11.000us      11.000us             1  
                                                 aten::abs_         0.18%       5.000us         0.37%      10.000us       5.000us             2  
                                           aten::empty_like         0.26%       7.000us         0.37%      10.000us       5.000us             2  
                                                aten::clamp         0.26%       7.000us         0.37%      10.000us      10.000us             1  
                                           aten::zeros_like         0.18%       5.000us         0.33%       9.000us       9.000us             1  
                                                   aten::eq         0.33%       9.000us         0.33%       9.000us       3.000us             3  
                                                aten::zero_         0.30%       8.000us         0.30%       8.000us       0.727us            11  
                                         aten::_unsafe_view         0.22%       6.000us         0.30%       8.000us       4.000us             2  
                                                aten::fill_         0.30%       8.000us         0.30%       8.000us       2.000us             4  
                                              aten::reshape         0.18%       5.000us         0.30%       8.000us       8.000us             1  
                                  aten::_local_scalar_dense         0.22%       6.000us         0.26%       7.000us       1.000us             7  
            autograd::engine::evaluate_function: TBackward0         0.11%       3.000us         0.26%       7.000us       7.000us             1  
                                              aten::resize_         0.22%       6.000us         0.22%       6.000us       1.200us             5  
                                           aten::reciprocal         0.22%       6.000us         0.22%       6.000us       6.000us             1  
                                                 aten::sgn_         0.15%       4.000us         0.15%       4.000us       4.000us             1  
                                                 TBackward0         0.07%       2.000us         0.15%       4.000us       4.000us             1  
                                       aten::_reshape_alias         0.11%       3.000us         0.11%       3.000us       3.000us             1  
                                            aten::clamp_max         0.11%       3.000us         0.11%       3.000us       3.000us             1  
                                         aten::resolve_conj         0.07%       2.000us         0.07%       2.000us       0.333us             6  
                                    aten::broadcast_tensors         0.07%       2.000us         0.07%       2.000us       1.000us             2  
                                               AddBackward0         0.04%       1.000us         0.04%       1.000us       1.000us             1  
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
    Self CPU time total: 2.711ms  <===  above: Self CPU time total: 18.451ms (... the impact of warm-up)
    """

    @scheduler
    @profile
    def train():
        # The step() function is executed inside speechbrain.core.brain.fit and is property of the Brain's profiler.
        # Above profiler and its scheduler are without power here, since prof.step() is not run - at all.
        brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)

    prof = train()
    # since we used the same brain (which has its own profiler)
    assert brain.profiler.step_num == 40
    assert len(brain.profiler.speechbrain_parsed_kineto_traces) == 2
    assert len(brain.profiler.events()) == 293   # unchanged (overwritten with akin data)
    assert len(brain.profiler.key_averages()) == 73   # unchanged (akin data)
    # now, to the train function's profiler
    assert prof.step_num == 0  # the prof.step() operation wasn't run (not in scope) -> its scheduler is unawaken!
    assert not hasattr(prof, "speechbrain_parsed_kineto_traces")  # no trace collection
    with raises(Exception) as err_prof:
        prof.events()  # No tracing started with this one.
    assert err_prof.type == AssertionError  # sparing: key_averages()

    # But how to add profiling then if no writing access is there for a class... pretrained, for example:
    class SimpleBrainUntracked(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    brain_or_pretrained = SimpleBrainUntracked({"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device})
    """no need to define an extra function
    @scheduler
    @profile
    def eval():
        brain.evaluate(test_set=test_set)
    prof = eval()
    """
    # brain_or_pretrained = scheduler(profile(brain_or_pretrained)) # todo
    # brain_or_pretrained.evaluate(test_set=test_set)
