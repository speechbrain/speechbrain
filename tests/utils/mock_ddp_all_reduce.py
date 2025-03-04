"""
Mock ddp_all_reduce, so that it becomes easy to test functions that use it,
without actually using threads or multiple processors.

Authors
 * Rogier van Dalen 2025
"""

from typing import Callable, List
from unittest.mock import patch

import torch


class NotFinishedYet(BaseException):
    pass


def run_with_mocked_ddp_all_reduce(functions: List[Callable]) -> list:
    """
    Run multiple functions while pretending that they are running on multiple
    threads and can exchange tensors using synchronised calls to ddp_all_reduce.

    Since this actually runs on a single thread, there are a few requirements:
    * The functions must call module.ddp_all_reduce and not naked
      ddp_all_reduce.
      This is because otherwise the function cannot be patched.
    * The functions must have deterministic behaviour, and this should not
      change if ddp_all_reduce raises an exception and the function is called
      again afterwards.
    * The functions must each call ddp_all_reduce the same number of times.
      This should be the case for correctness.
    * The functions must not catch all exceptions.
      This is because this function uses an exception for control flow.

    Arguments
    ---------
    functions: list[callable]
        The list of functions that communicate with each other using
        ddp_all_reduce.
        These are likely the same functions or the same functions with different
        arguments, for example using "lambda".
        The functions should take no argument.

    Returns
    -------
    return_values: list
        A list with the return values of each of the functions.
    """
    # This runs all the functions as many times as they call ddp_all_reduce.
    # This is because this function cannot run all the functions literally at
    # the same time, so it cancels each function call (by raising an exception)
    # for every call to ddp_all_reduce after noting down the function's
    # contribution.
    # So it first runs all the functions up to their first call to
    # ddp_all_reduce, then up to the second, et cetera.

    # The arguments that each function passes in to ddp_all_reduce.
    intermediate_arguments = [[] for _ in functions]
    # False until the function is done.
    done = [False for _ in functions]
    # Return values for each function.
    return_values = [None for _ in functions]

    def make_mock_all_reduce(function_index):
        # Count the number of calls.
        num_calls = [0]

        def mock_all_reduce(
            tensor: torch.Tensor, op: torch.distributed.ReduceOp
        ):
            call_index = num_calls[0]

            if call_index < len(intermediate_arguments[function_index]):
                # Keep going.
                previous_argument = intermediate_arguments[function_index][
                    call_index
                ]
                assert torch.allclose(previous_argument, tensor), (
                    "The function should pass in values deterministically "
                    "for this test to work"
                )
                # We should have seen the calls from the other functions too.
                assert all(
                    (call_index < len(arguments))
                    for arguments in intermediate_arguments
                )
                peer_arguments = torch.stack(
                    [
                        arguments[call_index]
                        for arguments in intermediate_arguments
                    ]
                )

                # Return (by setting the actual tensor, like ddp_reduce_all
                # does) the reduced value.
                assert (
                    op == torch.distributed.ReduceOp.SUM
                ), "Not supported in mock"
                result = torch.sum(peer_arguments, dim=0)
                tensor.set_(result)

                num_calls[0] += 1

            else:
                assert call_index == len(intermediate_arguments[function_index])
                # Save the argument passed in and cancel the function call.
                intermediate_arguments[function_index].append(tensor)
                raise NotFinishedYet

        return mock_all_reduce

    def run_function(index: int, function: Callable):
        try:
            mock = make_mock_all_reduce(index)
            with patch(
                "speechbrain.utils.distributed.ddp_all_reduce", new=mock
            ):
                result = function()

            # No exceptions raised: this function is done.
            done[index] = True
            return_values[index] = result
        except NotFinishedYet:
            pass

    while True:
        for index, function in enumerate(functions):
            run_function(index, function)
        if any(done):
            assert all(
                done
            ), "Not all functions call all_reduce the same number of times"
            return return_values
