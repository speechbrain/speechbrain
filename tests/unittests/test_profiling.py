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
    brain = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    assert brain.profiler is not None
    assert brain.profiler.profiler is not None
    # assert len(brain.profiler.key_averages()) == 2
    # assert (brain.profiler.events().total_average().count >= 4)  # == 6  # before; config dependent: 7
    assert (
        len(brain.profiler.speechbrain_event_traces) == 1
    )  # set & filled by the @profile decorator
    """print(brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
        aten::_has_compatible_shallow_copy_type        73.33%      11.000us        73.33%      11.000us       2.750us             4
                                       aten::to        26.67%       4.000us        26.67%       4.000us       2.000us             2
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 15.000us
    """

    # Profiling: fit() for train operations.
    # By default, @profile should also annotate fit & evaluate functions; here the fit function is tested only.
    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    assert brain.profiler is not None
    # assert len(brain.profiler.key_averages()) >= 60  # 72 with torch==1.10.1
    # assert brain.profiler.events().total_average().count >= 2000  # 2832 with torch==1.10.1
    assert len(brain.profiler.speechbrain_event_traces) == 2
    # assert len(brain.profiler.speechbrain_event_traces[0]) >= 4  # == 6  # before; config dependent: 7
    # assert len(brain.profiler.speechbrain_event_traces[1]) >= 2000  # 2862 with torch==1.10.1
    """print(brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::l1_loss         2.60%     443.000us        26.15%       4.460ms     111.500us            40
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        12.97%       2.212ms        20.28%       3.459ms      86.475us            40
                                               aten::linear         1.28%     219.000us        12.97%       2.212ms     110.600us            20
                                                   aten::to         1.88%     320.000us        10.68%       1.822ms      11.387us           160
                                             aten::isfinite         1.98%     338.000us         9.82%       1.674ms      55.800us            30
                                                 aten::mean         1.10%     188.000us         9.31%       1.587ms      79.350us            20
                                             aten::_to_copy         5.65%     964.000us         8.99%       1.533ms      15.330us           100
                                                aten::stack         2.23%     380.000us         8.67%       1.479ms      29.580us            50
                                               aten::matmul         1.62%     277.000us         7.63%       1.301ms      65.050us            20
                                     aten::l1_loss_backward         1.24%     212.000us         6.78%       1.157ms      57.850us            20
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 17.054ms
    """


def test_profile_func(device):
    # import torch
    # from pytest import raises
    # from torch.optim import SGD
    # from speechbrain.core import Brain
    # from torch.autograd.profiler import record_function
    from speechbrain.utils.profiling import profile

    # from speechbrain.utils.profiling import events_diff

    """
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

    @profile
    def train(brain, train_set, valid_set):
        brain.fit(
            epoch_counter=range(10), train_set=train_set, valid_set=valid_set
        )

    """
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
    # assert len(prof_simple.events()) >= 2500  # 2832 with torch==1.10.1
    # assert len(prof_simple.key_averages()) >= 60  # 72 with torch==1.10.1

    simple_brain_nitty_gritty = SimpleBrainNittyGritty(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    prof_nitty_gritty = train(
        simple_brain_nitty_gritty, training_set, validation_set
    )
    # print(prof_nitty_gritty.key_averages().table(sort_by="cpu_time_total"))
    # assert len(prof_nitty_gritty.events()) >= 2500  # 3030 with torch==1.10.1
    # assert len(prof_nitty_gritty.key_averages()) >= 60  # 74 with torch==1.10.1
    """

    # The outputs of this diff are only for visualisation, ``simple_delta._build_tree()`` will throw an error.
    """
    simple_delta, nitty_gritty_delta = events_diff(
        prof_simple.key_averages(), prof_nitty_gritty.key_averages()
    )
    # assert len(simple_delta) >= 4  # == 6  # before; config dependent: 7
    # assert len(nitty_gritty_delta) >= 4  # == 8  # before
    # assert simple_delta.total_average().count == 582 #Switching off becuase sometimes it fails
    # assert nitty_gritty_delta.total_average().count == 780 #Switching off becuase sometimes it fails
    with raises(Exception) as err_tree:
        simple_delta._build_tree()  # as mentioned.
    assert err_tree.type == AttributeError
    with raises(Exception) as err_averages:
        simple_delta.key_averages()  # as mentioned.
    assert err_averages.type == AssertionError
    " ""Both classes have alike numbers of function calls (given the same input data and train function).
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
    from speechbrain.utils.profiling import profile, schedule

    @schedule
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
    brain = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    assert brain.profiler.profiler is None
    assert len(brain.profiler.speechbrain_event_traces) == 0
    with raises(Exception) as err:
        brain.profiler.events()  # Tracing hasn't started, yet, so everything is in err. Scheduler says: wait.
    assert err.type == AssertionError
    assert brain.profiler.step_num == 0

    # Profiling: fit() for train operations.
    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    assert brain.profiler.step_num == 20
    assert len(brain.profiler.speechbrain_event_traces) == 1
    # assert len(brain.profiler.events()) >= 250  # 293 with torch==1.10.1
    # assert len(brain.profiler.key_averages()) >= 60  # 73 with torch==1.10.1
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

    @schedule
    @profile
    def train():
        # The step() function is executed inside speechbrain.core.brain.fit and is property of the Brain's profiler.
        # Above profiler and its scheduler are without power here, since prof.step() is not run - at all.
        brain.fit(
            epoch_counter=range(10), train_set=train_set, valid_set=valid_set
        )

    prof = train()
    # since we used the same brain (which has its own profiler)
    # assert brain.profiler.step_num == 20  # started again from 0 steps
    assert len(brain.profiler.speechbrain_event_traces) == 2
    # assert len(brain.profiler.events()) >= 250  # 293 with torch==1.10.1  # unchanged (overwritten with akin data)
    # assert len(brain.profiler.key_averages()) >= 60  # 73 with torch==1.10.1 # unchanged (akin data)
    # now, to the train function's profiler
    assert (
        prof.step_num == 0
    )  # the prof.step() operation wasn't run (not in scope) -> its scheduler is unawaken!
    assert not hasattr(prof, "speechbrain_event_traces")  # no trace collection
    with raises(Exception) as err_prof:
        prof.events()  # No tracing started with this one.
    assert err_prof.type == AssertionError  # sparing: key_averages()

    # But how to add profiling then if no writing access is there for a class... pretrained, for example:
    class SimpleBrainUntracked(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    brain_or_pretrained = SimpleBrainUntracked(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )

    # Set-up the profiler and hook it to the model.
    scheduled_profiler = schedule(profile)
    scheduled_profiler(brain_or_pretrained)

    # Profiling: still too early for scheduler!
    brain_or_pretrained.evaluate(test_set=test_set)  # -> step_num=1
    assert brain_or_pretrained.profiler.step_num == 1
    assert brain_or_pretrained.profiler.profiler is None

    # Profiling: scheduler warms-up.
    brain_or_pretrained.evaluate(
        test_set=(
            [inputs, targets],  # +1x test_set -> step_num=2
            [inputs, targets],
        )
    )
    assert brain_or_pretrained.profiler.step_num == 2
    # brain_or_pretrained.profiler.profiler will be set (not None anymore)
    # when run on cpu, there are no events - but cuda activities are recorded if existing
    # see: https://github.com/speechbrain/speechbrain/issues/1469
    if (
        torch.profiler.ProfilerActivity.CUDA
        in brain_or_pretrained.profiler.activities
    ):
        assert (
            len(
                set(
                    [
                        x.name
                        for x in brain_or_pretrained.profiler.profiler.function_events
                    ]
                )
                - {
                    "cudaGetDeviceCount",
                    "cudaGetDeviceProperties",
                    "cudaDeviceSynchronize",
                }
            )
        ) == 0
    else:
        assert len(brain_or_pretrained.profiler.events()) == 0

    # Profiling: scheduler warms-up...
    brain_or_pretrained.evaluate(
        test_set=(
            [inputs, targets],  # +1x test_set
            [inputs, targets],  # +2x test_set -> step_num=3
            [inputs, targets],
        )
    )
    assert brain_or_pretrained.profiler.step_num == 3
    if (
        torch.profiler.ProfilerActivity.CUDA
        in brain_or_pretrained.profiler.activities
    ):
        assert (
            len(
                set(
                    [
                        x.name
                        for x in brain_or_pretrained.profiler.profiler.function_events
                    ]
                )
                - {
                    "cudaGetDeviceCount",
                    "cudaGetDeviceProperties",
                    "cudaDeviceSynchronize",
                }
            )
        ) == 0
    else:
        assert len(brain_or_pretrained.profiler.events()) == 0

    # Profiling: first trace!
    brain_or_pretrained.evaluate(
        test_set=(
            [inputs, targets],  # +1x test_set
            [inputs, targets],  # +2x test_set
            [inputs, targets],  # +3x test_set -> step_num=4
            [inputs, targets],
        )
    )
    assert brain_or_pretrained.profiler.step_num == 4
    # assert len(brain_or_pretrained.profiler.events()) >= 4  # == 10  # before
    # assert len(brain_or_pretrained.profiler.key_averages()) >= 4  # == 5  # before
    assert (
        len(brain_or_pretrained.profiler.events()) >= 1
    )  # 1 on CPU; more w/ CUDA


def test_tracer(device):
    import torch
    from torch.optim import SGD
    from speechbrain.core import Brain
    from speechbrain.utils.profiling import profile, export

    @export
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

    # Profiling: __init__ constructor and model training.
    brain = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)

    # Pretrained example.
    class SimpleBrainUntracked(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    # No tracing during __init__
    brain_or_pretrained = SimpleBrainUntracked(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    profile(brain_or_pretrained, on_trace_ready=export(), with_stack=True)
    brain_or_pretrained.evaluate(test_set=test_set)

    # Set-up the profiler; hook it to the model, and benchmark inference.
    brain_or_pretrained2 = SimpleBrainUntracked(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    logged_profiler = export(profile)
    assert brain_or_pretrained2.profiler is None
    logged_profiler(brain_or_pretrained2)
    brain_or_pretrained2.evaluate(test_set=test_set)


def test_aggregated_traces(device):
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
    test_set = (
        [inputs, targets],
        [inputs, targets],
    )

    # Profiling: __init__ constructor -- while scheduler: waiting --> nothing to report
    brain = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )

    # Profiling: empty traces
    assert len(brain.profiler.speechbrain_event_traces) == 1
    """
    init_report = brain.profiler.merge_traces()
    assert len(init_report) >= 1
    # assert len(init_report) >= 4  # == 6  # before; config dependent: 7
    assert len(brain.profiler.speechbrain_event_traces) == 1
    " ""print(brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
        aten::_has_compatible_shallow_copy_type        80.00%      12.000us        80.00%      12.000us       3.000us             4
                                       aten::to        20.00%       3.000us        20.00%       3.000us       1.500us             2
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 15.000us

    print(init_report.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                           Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
        aten::_has_compatible_shallow_copy_type        80.00%      12.000us        80.00%      12.000us       3.000us             4
                                       aten::to        20.00%       3.000us        20.00%       3.000us       1.500us             2
    -------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 15.000us
    """

    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    assert len(brain.profiler.speechbrain_event_traces) == 2
    # assert len(brain.profiler.speechbrain_event_traces[0]) >= 4  # == 6  # before; config dependent: 7
    # assert len(brain.profiler.speechbrain_event_traces[1]) >= 2500  # 2862 with torch==1.10.1
    # assert len(brain.profiler.events()) >= 2500  # 2832 with torch==1.10.1
    # assert len(brain.profiler.events().key_averages()) >= 60  # 72 with torch==1.10.1
    """print(brain.profiler.events().key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::l1_loss         2.39%     415.000us        25.28%       4.392ms     109.800us            40
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        12.65%       2.198ms        20.06%       3.485ms      87.125us            40
                                               aten::linear         1.41%     245.000us        13.24%       2.299ms     114.950us            20
                                                   aten::to         2.04%     354.000us        10.59%       1.839ms      11.494us           160
                                             aten::isfinite         2.12%     369.000us         9.87%       1.714ms      57.133us            30
                                                 aten::mean         1.13%     196.000us         9.15%       1.589ms      79.450us            20
                                                aten::stack         2.33%     404.000us         8.94%       1.553ms      31.060us            50
                                             aten::_to_copy         5.67%     985.000us         8.67%       1.506ms      15.060us           100
                                               aten::matmul         1.57%     273.000us         7.83%       1.360ms      68.000us            20
                                     aten::l1_loss_backward         1.22%     212.000us         6.67%       1.158ms      57.900us            20
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 17.370ms
    """

    # Profiling: aggregate traces
    """
    short_report = brain.profiler.merge_traces()
    assert len(short_report) >= 1
    # assert len(short_report) >= 2500  # 2838 with torch==1.10.1
    # assert len(short_report.key_averages()) >= 60  # 73 with torch==1.10.1
    " ""print(short_report.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::l1_loss         2.39%     415.000us        25.26%       4.392ms     109.800us            40
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        12.64%       2.198ms        20.05%       3.485ms      87.125us            40
                                               aten::linear         1.41%     245.000us        13.22%       2.299ms     114.950us            20
                                                   aten::to         2.05%     357.000us        10.60%       1.842ms      11.370us           162
                                             aten::isfinite         2.12%     369.000us         9.86%       1.714ms      57.133us            30
                                                 aten::mean         1.13%     196.000us         9.14%       1.589ms      79.450us            20
                                                aten::stack         2.32%     404.000us         8.93%       1.553ms      31.060us            50
                                             aten::_to_copy         5.67%     985.000us         8.66%       1.506ms      15.060us           100
                                               aten::matmul         1.57%     273.000us         7.82%       1.360ms      68.000us            20
                                     aten::l1_loss_backward         1.22%     212.000us         6.66%       1.158ms      57.900us            20
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 17.385ms
    """

    brain.evaluate(test_set=test_set)
    brain.evaluate(test_set=test_set)
    brain.evaluate(test_set=test_set)
    assert len(brain.profiler.speechbrain_event_traces) == 5
    # assert len(brain.profiler.speechbrain_event_traces[0]) >= 4  # == 6  # before; config dependent: 7
    # assert len(brain.profiler.speechbrain_event_traces[1]) >= 2500  # 2862 with torch==1.10.1
    # assert len(brain.profiler.speechbrain_event_traces[2]) >= 125  # 143 with torch==1.10.1
    # assert len(brain.profiler.speechbrain_event_traces[3]) >= 125  # 143 with torch==1.10.1
    # assert len(brain.profiler.speechbrain_event_traces[4]) >= 125  # 143 with torch==1.10.1
    # assert len(brain.profiler.events()) >= 125  # 141 with torch==1.10.1
    # assert len(brain.profiler.events().key_averages()) >= 25  # 42 with torch==1.10.1
    # the following is only for the last call of the 3x brain.evaluate()
    """print(brain.profiler.events().key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::l1_loss         3.54%      23.000us        37.38%     243.000us      60.750us             4
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        16.62%     108.000us        29.38%     191.000us      63.667us             3
                                               aten::linear         2.77%      18.000us        20.46%     133.000us      66.500us             2
                                             aten::isfinite         3.54%      23.000us        14.31%      93.000us      46.500us             2
                                                 aten::mean         1.85%      12.000us        12.62%      82.000us      41.000us             2
                                                aten::stack         3.23%      21.000us        12.15%      79.000us      19.750us             4
                                               aten::matmul         2.62%      17.000us        11.69%      76.000us      38.000us             2
                                                 aten::div_         2.92%      19.000us         7.54%      49.000us      24.500us             2
                                                   aten::to         1.85%      12.000us         7.08%      46.000us       7.667us             6
                                                   aten::mm         6.31%      41.000us         6.77%      44.000us      22.000us             2
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 650.000us
    """

    # Profiling: putting previous benchmark reporting together.
    """
    full_report = brain.profiler.merge_traces()
    assert len(full_report) >= 1
    # assert len(full_report.key_averages()) >= 60  # 73 with torch==1.10.1
    # In this minimal example, only 73 functions matter.
    # Some events are duplicated (perhaps from wrapping functions):
    # => they appear stacked & EventList._remove_dup_nodes drops direct child events of same name as their parent.
    num_events = sum([len(x) for x in brain.profiler.speechbrain_event_traces])
    assert num_events >= 1
    # assert num_events >= 3000  # 3297 with torch==1.10.1  # expected: 6 + 2862 + 3x143 = 3297
    # Apparently, this depends on how this test is run (by its own or as part of the entire file's test suite).
    # assert (num_events == len(full_report)) or (len(full_report) == len(set([x.id for x in full_report])))
    # ... not tested, why
    " ""print(full_report.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              aten::l1_loss         2.21%     427.000us        27.58%       5.326ms     102.423us            52
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        11.89%       2.297ms        21.80%       4.210ms      85.918us            49
                                               aten::linear         1.57%     304.000us        14.61%       2.821ms     108.500us            26
                                             aten::isfinite         2.53%     488.000us        10.72%       2.071ms      57.528us            36
                                                   aten::to         2.03%     392.000us        10.42%       2.013ms      11.183us           180
                                                 aten::mean         1.15%     223.000us         9.81%       1.894ms      72.846us            26
                                                aten::stack         2.34%     452.000us         9.54%       1.842ms      29.710us            62
                                             aten::_to_copy         5.49%       1.061ms         8.51%       1.643ms      14.670us           112
                                               aten::matmul         1.65%     318.000us         8.48%       1.638ms      63.000us            26
                                                 aten::div_         1.50%     290.000us         6.95%       1.343ms      47.964us            28
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 19.311ms
    """
    # 19.311ms = 3x ~650.000us + 17.370ms + 15.000us <=> 1st & 2nd call of brain.evaluate() = 1276us = 2x 638us
    # max([x.time_range.end for x in full_report]) -> 41965 (us)


def test_profile_details(device):
    import torch

    # from copy import deepcopy
    from torch.optim import SGD
    from speechbrain.core import Brain
    from speechbrain.utils.profiling import (
        profile_analyst,
        profile_optimiser,
        export,
        # events_diff,
    )

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
    test_set = (
        [inputs, targets],
        [inputs, targets],
        [inputs, targets],
        [inputs, targets],
        [inputs, targets],
        [inputs, targets],
    )

    brain_analyst = profile_analyst(
        SimpleBrain(
            {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
        )
    )
    brain_optimiser = profile_optimiser(
        SimpleBrain(
            {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
        )
    )

    assert len(brain_analyst.profiler.speechbrain_event_traces) == 0
    brain_analyst.fit(
        epoch_counter=range(10), train_set=train_set, valid_set=valid_set
    )
    assert len(brain_analyst.profiler.speechbrain_event_traces) == 1
    # assert len(brain_analyst.profiler.speechbrain_event_traces[0]) >= 250  # 296 with torch==1.10.1
    # assert len(brain_analyst.profiler.events()) >= 250  # 293 with torch==1.10.1
    # assert len(brain_analyst.profiler.events().key_averages()) >= 60  # 73 with torch==1.10.1
    """print(brain_analyst.profiler.events().key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls   Total FLOPs
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              ProfilerStep*        36.62%       4.345ms        98.50%      11.686ms       5.843ms         796 b      -1.66 Kb             2            --
                                              aten::l1_loss         2.50%     297.000us        14.37%       1.705ms     426.250us          16 b        -800 b             4            --
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...         3.91%     464.000us        11.01%       1.306ms     326.500us       1.55 Kb         -80 b             4            --
                                             aten::isfinite         3.10%     368.000us         9.65%       1.145ms     381.667us           3 b         -18 b             3            --
                                                aten::stack         2.79%     331.000us         8.46%       1.004ms     200.800us       1.57 Kb           0 b             5            --
                                                   aten::to         1.96%     232.000us         7.25%     860.000us      53.750us          40 b           0 b            16            --
                                               aten::linear         1.40%     166.000us         6.55%     777.000us     388.500us         800 b           0 b             2            --
                                     aten::l1_loss_backward         1.58%     188.000us         6.08%     721.000us     360.500us         400 b          -4 b             2            --
                                             aten::_to_copy         4.28%     508.000us         5.29%     628.000us      62.800us          40 b           0 b            10            --
                                                 aten::mean         0.89%     105.000us         4.44%     527.000us     263.500us           8 b           8 b             2            --
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 11.864ms
    """
    # 6-batch inference
    brain_analyst.evaluate(test_set=test_set)
    assert len(brain_analyst.profiler.speechbrain_event_traces) == 2
    # assert len(brain_analyst.profiler.speechbrain_event_traces[0]) >= 250  # 296 with torch==1.10.1
    # assert len(brain_analyst.profiler.speechbrain_event_traces[1]) >= 125  # 144 with torch==1.10.1
    # as of evaluate() call
    # assert len(brain_analyst.profiler.events()) >= 125  # 142 with torch==1.10.1
    # assert len(brain_analyst.profiler.events().key_averages()) >= 25  # 42 with torch==1.10.1
    """print(brain_analyst.profiler.events().key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls   Total FLOPs
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              ProfilerStep*        19.24%       1.129ms        96.92%       5.687ms       2.844ms         796 b      -1.61 Kb             2            --
                                              aten::l1_loss         5.16%     303.000us        35.50%       2.083ms     520.750us          16 b        -800 b             4            --
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...         7.41%     435.000us        25.95%       1.523ms     761.500us       1.55 Kb         -40 b             2            --
                                                aten::stack         6.15%     361.000us        18.20%       1.068ms     267.000us       1.56 Kb           0 b             4            --
                                               aten::linear         3.43%     201.000us        15.78%     926.000us     463.000us         800 b           0 b             2            --
                                                 aten::mean         2.42%     142.000us        11.93%     700.000us     350.000us           8 b           8 b             2            --
                                             aten::isfinite         3.72%     218.000us        10.84%     636.000us     318.000us           2 b         -12 b             2            --
                                                  aten::cat         2.68%     157.000us         8.95%     525.000us     131.250us       1.56 Kb           0 b             4            --
                                               aten::matmul         3.83%     225.000us         8.88%     521.000us     260.500us         800 b           0 b             2            --
                                                 aten::div_         3.34%     196.000us         6.97%     409.000us     204.500us           0 b          -8 b             2            --
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 5.868ms
    """

    brain_optimiser.fit(
        epoch_counter=range(10), train_set=train_set, valid_set=valid_set
    )
    # key_avg_fit = deepcopy(brain_optimiser.profiler.events().key_averages())
    """print(brain_optimiser.profiler.events().key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              ProfilerStep*        50.73%       1.874ms        98.86%       3.652ms       1.826ms         796 b      -1.66 Kb             2
                                              aten::l1_loss         1.38%      51.000us        11.83%     437.000us     109.250us          16 b        -400 b             4
                                             aten::isfinite         1.49%      55.000us         8.69%     321.000us     107.000us           3 b         -16 b             3
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...         4.03%     149.000us         7.58%     280.000us      70.000us       1.55 Kb         -64 b             4
                                               aten::linear         0.51%      19.000us         5.66%     209.000us     104.500us         800 b           0 b             2
                                                  aten::abs         3.19%     118.000us         5.58%     206.000us      25.750us          24 b          12 b             8
                                     aten::l1_loss_backward         0.92%      34.000us         4.28%     158.000us      79.000us         400 b          -4 b             2
                                                aten::stack         1.00%      37.000us         3.87%     143.000us      28.600us       1.57 Kb           0 b             5
                                                aten::empty         3.76%     139.000us         3.76%     139.000us       4.793us         544 b         544 b            29
                                                   aten::to         0.76%      28.000us         3.76%     139.000us       8.688us          44 b           4 b            16
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 3.694ms
    # to 11.864ms (analyst)
    """

    brain_optimiser.evaluate(test_set=test_set)
    """
    key_avg_evaluate = deepcopy(
        brain_optimiser.profiler.events().key_averages()
    )
    """
    """print(brain_optimiser.profiler.events().key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              ProfilerStep*        24.80%     524.000us        96.50%       2.039ms       1.020ms         796 b      -1.61 Kb             2
                                              aten::l1_loss         2.74%      58.000us        33.65%     711.000us     177.750us          16 b        -800 b             4
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...         9.94%     210.000us        21.11%     446.000us     223.000us       1.55 Kb         -40 b             2
                                             aten::isfinite         3.64%      77.000us        15.76%     333.000us     166.500us           2 b         -12 b             2
                                               aten::linear         1.04%      22.000us        11.88%     251.000us     125.500us         800 b           0 b             2
                                                 aten::mean         2.04%      43.000us        11.74%     248.000us     124.000us           8 b           8 b             2
                                                aten::stack         3.31%      70.000us        10.18%     215.000us      53.750us       1.56 Kb           0 b             4
                                                 aten::div_         4.83%     102.000us         7.90%     167.000us      83.500us           0 b          -8 b             2
                                               aten::matmul         1.61%      34.000us         7.38%     156.000us      78.000us         800 b           0 b             2
                                                   aten::ne         4.54%      96.000us         6.72%     142.000us      71.000us           2 b          -6 b             2
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 2.113ms
    # to 5.868ms (analyst)
    """
    # same check as for analyst
    assert len(brain_optimiser.profiler.speechbrain_event_traces) == 2
    # assert len(brain_optimiser.profiler.speechbrain_event_traces[0]) >= 250  # 296 with torch==1.10.1
    # assert len(brain_optimiser.profiler.speechbrain_event_traces[1]) >= 125  # 144 with torch==1.10.1
    # as of evaluate() call
    # assert len(brain_optimiser.profiler.events()) >= 125  # 142 with torch==1.10.1
    # assert len(brain_optimiser.profiler.events().key_averages()) >= 25  # 42 with torch==1.10.1
    # different config
    assert (
        brain_optimiser.profiler.record_shapes
        != brain_analyst.profiler.record_shapes
    )
    assert (
        brain_optimiser.profiler.with_stack != brain_analyst.profiler.with_stack
    )
    assert (
        brain_optimiser.profiler.with_flops != brain_analyst.profiler.with_flops
    )
    # same config
    assert (
        brain_optimiser.profiler.with_modules
        == brain_analyst.profiler.with_modules
    )
    assert (
        brain_optimiser.profiler.profile_memory
        == brain_analyst.profiler.profile_memory
    )

    """
    # let's take a look at the diff
    diff_fit, diff_evaluate = events_diff(key_avg_fit, key_avg_evaluate)
    # assert len(diff_fit) >= 50  # 64 with torch==1.10.1
    # assert len(diff_evaluate) >= 25  # 33 with torch==1.10.1
    # assert diff_fit.total_average().count >= 250  # 273 with torch==1.10.1
    # assert diff_evaluate.total_average().count >= 100  # 122 with torch==1.10.1
    " ""For curiosity only... the printed FunctionEvents differ by (name, # of Calls)
    print(diff_fit.table(sort_by="cpu_time_total"))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                             aten::isfinite         3.35%      55.000us        19.55%     321.000us     107.000us           3 b         -16 b             3
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...         9.07%     149.000us        17.05%     280.000us      70.000us       1.55 Kb         -64 b             4
                                                  aten::abs         7.19%     118.000us        12.55%     206.000us      25.750us          24 b          12 b             8
                                     aten::l1_loss_backward         2.07%      34.000us         9.62%     158.000us      79.000us         400 b          -4 b             2
                                                aten::stack         2.25%      37.000us         8.71%     143.000us      28.600us       1.57 Kb           0 b             5
                                                aten::empty         8.47%     139.000us         8.47%     139.000us       4.793us         544 b         544 b            29
                                                   aten::to         1.71%      28.000us         8.47%     139.000us       8.688us          44 b           4 b            16
                                             aten::_to_copy         1.89%      31.000us         7.00%     115.000us      11.500us          40 b           0 b            10
                                                   aten::mm         6.46%     106.000us         6.58%     108.000us      36.000us       1.17 Kb       1.17 Kb             3
                                                   aten::ne         4.26%      70.000us         6.46%     106.000us      35.333us           3 b          -9 b             3
                                                aten::zeros         1.95%      32.000us         6.03%      99.000us      12.375us          32 b           0 b             8
    autograd::engine::evaluate_function: L1LossBackward0...         0.37%       6.000us         5.97%      98.000us      98.000us         396 b          -4 b             1
                                            L1LossBackward0         0.24%       4.000us         5.60%      92.000us      92.000us         400 b           0 b             1
                                                  aten::cat         0.97%      16.000us         5.05%      83.000us      16.600us       1.57 Kb           0 b             5
                                                  aten::div         2.80%      46.000us         4.81%      79.000us      26.333us          12 b           0 b             3
                                    Optimizer.step#SGD.step         3.71%      61.000us         4.51%      74.000us      74.000us          -4 b         -20 b             1
                                                 aten::_cat         2.13%      35.000us         4.08%      67.000us      13.400us       1.57 Kb           0 b             5
                                        aten::empty_strided         3.41%      56.000us         3.41%      56.000us       5.091us          44 b          44 b            11
           autograd::engine::evaluate_function: MmBackward0         0.37%       6.000us         3.41%      56.000us      56.000us           0 b        -400 b             1
                                                    aten::t         2.13%      35.000us         3.11%      51.000us      10.200us           0 b           0 b             5
                                                 aten::add_         3.05%      50.000us         3.05%      50.000us       8.333us           0 b           0 b             6
                                                MmBackward0         0.43%       7.000us         3.05%      50.000us      50.000us         400 b           0 b             1
                                                  aten::mul         2.01%      33.000us         2.62%      43.000us      10.750us           5 b          -3 b             4
                                                  aten::sum         2.01%      33.000us         2.50%      41.000us      13.667us          40 b          40 b             3
          autograd::engine::evaluate_function: DivBackward0         0.55%       9.000us         2.38%      39.000us      39.000us          -4 b          -8 b             1
                                                 aten::norm         2.13%      35.000us         2.38%      39.000us      13.000us          12 b          12 b             3
                          Optimizer.zero_grad#SGD.zero_grad         1.46%      24.000us         1.89%      31.000us      31.000us          -4 b          -4 b             1
                                               DivBackward0         0.24%       4.000us         1.83%      30.000us      30.000us           4 b           0 b             1
                                                 aten::item         1.04%      17.000us         1.77%      29.000us       4.143us           0 b           0 b             7
          autograd::engine::evaluate_function: AddBackward0         0.43%       7.000us         1.77%      29.000us      29.000us          40 b           0 b             1
                                              aten::random_         1.71%      28.000us         1.71%      28.000us      14.000us           0 b           0 b             2
                                                  aten::sub         1.71%      28.000us         1.71%      28.000us       9.333us         800 b         800 b             3
                                                aten::copy_         1.71%      28.000us         1.71%      28.000us       2.800us           0 b           0 b            10
    autograd::engine::evaluate_function: torch::autograd...         0.30%       5.000us         1.71%      28.000us      14.000us        -440 b           0 b             2
                                              aten::resize_         1.64%      27.000us         1.64%      27.000us       5.400us       1.57 Kb       1.57 Kb             5
                                                  aten::add         0.85%      14.000us         1.64%      27.000us      27.000us           4 b           0 b             1
                                                   aten::eq         1.58%      26.000us         1.58%      26.000us       8.667us           3 b           3 b             3
                                            aten::unsqueeze         0.91%      15.000us         1.40%      23.000us       3.833us           0 b           0 b             6
                            torch::autograd::AccumulateGrad         0.97%      16.000us         1.40%      23.000us      11.500us        -440 b        -440 b             2
                                           aten::is_nonzero         0.43%       7.000us         1.40%      23.000us       7.667us           0 b           0 b             3
                                               aten::detach         0.55%       9.000us         1.28%      21.000us       3.500us           0 b           0 b             6
                                           aten::as_strided         1.16%      19.000us         1.16%      19.000us       1.188us           0 b           0 b            16
                                                 aten::view         1.04%      17.000us         1.04%      17.000us       3.400us           0 b           0 b             5
                                            aten::transpose         0.67%      11.000us         0.97%      16.000us       3.200us           0 b           0 b             5
                                           aten::empty_like         0.37%       6.000us         0.91%      15.000us       7.500us         404 b           0 b             2
                                                     detach         0.73%      12.000us         0.85%      14.000us       2.333us           0 b           0 b             6
                                                aten::clamp         0.61%      10.000us         0.85%      14.000us      14.000us           4 b           4 b             1
                                  aten::_local_scalar_dense         0.73%      12.000us         0.79%      13.000us       1.857us           0 b           0 b             7
                                                 aten::mul_         0.79%      13.000us         0.79%      13.000us       4.333us           0 b           0 b             3
                                            aten::ones_like         0.24%       4.000us         0.73%      12.000us      12.000us           4 b           0 b             1
                                           aten::zeros_like         0.30%       5.000us         0.73%      12.000us      12.000us         400 b           0 b             1
    autograd::engine::evaluate_function: UnsafeViewBackw...         0.18%       3.000us         0.73%      12.000us      12.000us           0 b           0 b             1
                                        UnsafeViewBackward0         0.12%       2.000us         0.55%       9.000us       9.000us           0 b           0 b             1
                                           aten::reciprocal         0.49%       8.000us         0.49%       8.000us       8.000us           4 b           4 b             1
                                              aten::reshape         0.24%       4.000us         0.43%       7.000us       7.000us           0 b           0 b             1
            autograd::engine::evaluate_function: TBackward0         0.12%       2.000us         0.43%       7.000us       7.000us           0 b           0 b             1
                                                aten::zero_         0.37%       6.000us         0.37%       6.000us       0.545us           0 b           0 b            11
                                                aten::fill_         0.37%       6.000us         0.37%       6.000us       1.500us           0 b           0 b             4
                                                 aten::sgn_         0.30%       5.000us         0.30%       5.000us       5.000us           0 b           0 b             1
                                                 TBackward0         0.06%       1.000us         0.30%       5.000us       5.000us           0 b           0 b             1
                                            aten::clamp_max         0.24%       4.000us         0.24%       4.000us       4.000us           0 b           0 b             1
                                       aten::_reshape_alias         0.18%       3.000us         0.18%       3.000us       3.000us           0 b           0 b             1
                                         aten::resolve_conj         0.12%       2.000us         0.12%       2.000us       0.333us           0 b           0 b             6
                                               AddBackward0         0.06%       1.000us         0.06%       1.000us       1.000us           0 b           0 b             1
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 1.642ms


    print(diff_evaluate.table(sort_by="cpu_time_total"))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    enumerate(DataLoader)#_SingleProcessDataLoaderIter._...        16.03%     210.000us        34.05%     446.000us     223.000us       1.55 Kb         -40 b             2
                                             aten::isfinite         5.88%      77.000us        25.42%     333.000us     166.500us           2 b         -12 b             2
                                                aten::stack         5.34%      70.000us        16.41%     215.000us      53.750us       1.56 Kb           0 b             4
                                                   aten::ne         7.33%      96.000us        10.84%     142.000us      71.000us           2 b          -6 b             2
                                                aten::empty        10.08%     132.000us        10.08%     132.000us       8.250us          80 b          80 b            16
                                                   aten::to         1.37%      18.000us         8.63%     113.000us      18.833us          16 b           0 b             6
                                                aten::zeros         3.13%      41.000us         8.55%     112.000us      28.000us          16 b           0 b             4
                                                  aten::cat         1.45%      19.000us         8.40%     110.000us      27.500us       1.56 Kb           0 b             4
                                                   aten::mm         7.02%      92.000us         7.33%      96.000us      48.000us         800 b         800 b             2
                                                  aten::abs         4.58%      60.000us         7.33%      96.000us      16.000us          16 b           8 b             6
                                             aten::_to_copy         2.21%      29.000us         7.25%      95.000us      23.750us          16 b           0 b             4
                                                 aten::_cat         2.82%      37.000us         6.95%      91.000us      22.750us       1.56 Kb           0 b             4
                                              aten::resize_         3.51%      46.000us         3.51%      46.000us      11.500us       1.56 Kb       1.56 Kb             4
                                        aten::empty_strided         3.36%      44.000us         3.36%      44.000us      11.000us          16 b          16 b             4
                                                  aten::sub         2.98%      39.000us         2.98%      39.000us      19.500us         800 b         800 b             2
                                                  aten::sum         2.14%      28.000us         2.90%      38.000us      19.000us           0 b           0 b             2
                                                 aten::add_         2.82%      37.000us         2.82%      37.000us      18.500us           0 b           0 b             2
                                                    aten::t         1.68%      22.000us         2.75%      36.000us      18.000us           0 b           0 b             2
                                            aten::unsqueeze         1.98%      26.000us         2.67%      35.000us       8.750us           0 b           0 b             4
                                                   aten::eq         2.37%      31.000us         2.37%      31.000us      15.500us           2 b           2 b             2
                                                  aten::mul         2.37%      31.000us         2.37%      31.000us      15.500us           2 b           2 b             2
                                           aten::is_nonzero         0.53%       7.000us         1.76%      23.000us      11.500us           0 b           0 b             2
                                                 aten::item         0.92%      12.000us         1.76%      23.000us       5.750us           0 b           0 b             4
                                                aten::copy_         1.68%      22.000us         1.68%      22.000us       5.500us           0 b           0 b             4
                                           aten::as_strided         1.37%      18.000us         1.37%      18.000us       2.250us           0 b           0 b             8
                                                 aten::view         1.30%      17.000us         1.30%      17.000us       4.250us           0 b           0 b             4
                                               aten::detach         0.31%       4.000us         1.15%      15.000us       7.500us           0 b           0 b             2
                                            aten::transpose         0.69%       9.000us         1.07%      14.000us       7.000us           0 b           0 b             2
                                                     detach         0.84%      11.000us         0.84%      11.000us       5.500us           0 b           0 b             2
                                  aten::_local_scalar_dense         0.84%      11.000us         0.84%      11.000us       2.750us           0 b           0 b             4
                                                aten::fill_         0.46%       6.000us         0.46%       6.000us       3.000us           0 b           0 b             2
                                                aten::zero_         0.31%       4.000us         0.31%       4.000us       1.000us           0 b           0 b             4
                                         aten::resolve_conj         0.31%       4.000us         0.31%       4.000us       1.000us           0 b           0 b             4
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 1.310ms
    """

    # set hook afterwards
    brain_analyst_raw = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    brain_optimiser_raw = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    brain_analyst_raw.fit(
        epoch_counter=range(10), train_set=train_set, valid_set=valid_set
    )
    profile_analyst(brain_analyst_raw)
    brain_analyst_raw.evaluate(test_set=test_set)
    assert getattr(brain_analyst_raw.profiler, "record_shapes") is True
    assert getattr(brain_analyst_raw.profiler, "with_stack") is True
    assert getattr(brain_analyst_raw.profiler, "with_flops") is True

    brain_optimiser_raw.fit(
        epoch_counter=range(10), train_set=train_set, valid_set=valid_set
    )
    profile_optimiser(brain_optimiser_raw)
    brain_optimiser_raw.evaluate(test_set=test_set)
    assert getattr(brain_optimiser_raw.profiler, "record_shapes") is False
    assert getattr(brain_optimiser_raw.profiler, "with_stack") is False
    assert getattr(brain_optimiser_raw.profiler, "with_flops") is False

    # wrap functions
    @profile_analyst
    def train_analyst(brain: SimpleBrain):
        brain.fit(
            epoch_counter=range(10), train_set=train_set, valid_set=valid_set
        )

    @export
    @profile_optimiser
    def evaluate_optimiser(brain: SimpleBrain):
        brain.evaluate(test_set=test_set)

    brain_raw = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    assert brain_raw.profiler is None
    train_analyst(brain_raw)
    assert brain_raw.profiler is None
    evaluate_optimiser(brain_raw)
    assert brain_raw.profiler is None

    # profile classes
    @export
    @profile_analyst
    class SimpleBrainAnalyst(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    @profile_optimiser
    class SimpleBrainOptimiser(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    simple_brain_analyst = SimpleBrainAnalyst(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    assert getattr(simple_brain_analyst.profiler, "record_shapes") is True
    assert getattr(simple_brain_analyst.profiler, "with_stack") is True
    assert getattr(simple_brain_analyst.profiler, "with_flops") is True
    simple_brain_analyst.evaluate(test_set=test_set)

    simple_brain_optimiser = SimpleBrainOptimiser(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )
    assert getattr(simple_brain_optimiser.profiler, "record_shapes") is False
    assert getattr(simple_brain_optimiser.profiler, "with_stack") is False
    assert getattr(simple_brain_optimiser.profiler, "with_flops") is False
    simple_brain_optimiser.fit(
        epoch_counter=range(10), train_set=train_set, valid_set=valid_set
    )
