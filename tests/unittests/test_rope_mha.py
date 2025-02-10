import pytest
import torch


@pytest.mark.skip("TODO make this work and switch back on")
def test_rope_torch_vs_homemade(device):
    """Test whether the output, loss and gradients are the same for custom MHA
    vs PyTorch MHSA with RoPe.

    Default rtol and atol values for the assert close are 1.3e-6, 1e-5 for fp32 and 1e-7 for fp64. Fp16 cannot be tested due to casting issues.
    """
    from speechbrain.nnet.attention import RoPEMHA, RoPEPytorchMHA

    dim_test = [16, 32, 64, 1024]
    num_heads = [1, 4, 8]

    for precision in [torch.float32, torch.float64]:
        for num_head in num_heads:
            for value in dim_test:
                inputs = torch.rand((4, 60, value)).to(
                    device=device, dtype=precision
                )
                torch.manual_seed(666)
                net = RoPEMHA(
                    num_heads=num_head, embed_dim=inputs.shape[-1]
                ).to(device=device, dtype=precision)

                torch.manual_seed(666)
                net2 = RoPEPytorchMHA(
                    num_heads=num_head, embed_dim=inputs.shape[-1]
                ).to(device=device, dtype=precision)

                outputs, attn = net(inputs, inputs, inputs)
                outputs2, attn2 = net2(inputs, inputs, inputs)

                torch.testing.assert_close(outputs, outputs2)

                labels = torch.rand((4, 60, value)).to(
                    device=device, dtype=precision
                )
                mse = torch.nn.MSELoss()
                loss1 = mse(outputs, labels)
                loss2 = mse(outputs2, labels)

                torch.testing.assert_close(loss1, loss2)

                loss1.backward()
                loss2.backward()

                sum1 = 0.0
                for p in net.parameters():
                    if p.requires_grad and p is not None:
                        sum1 += torch.sum(p.grad)
                sum2 = 0.0

                for p in net2.parameters():
                    if p.requires_grad and p is not None:
                        sum2 += torch.sum(p.grad)

                torch.testing.assert_close(sum1, sum2)
