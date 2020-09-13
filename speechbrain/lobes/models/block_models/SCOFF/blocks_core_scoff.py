import torch
import torch.nn as nn
from torch.distributions import normal

from speechbrain.lobes.models.block_models.utilities.BlockGRU import (
    BlockGRU,
    SharedBlockGRU,
)
from speechbrain.lobes.models.block_models.utilities.BlockLSTM import (
    BlockLSTM,
    SharedBlockLSTM,
)
from speechbrain.lobes.models.block_models.utilities.attention import (
    MultiHeadAttention,
)
from speechbrain.lobes.models.block_models.utilities.sparse_grad_attn import (
    blocked_grad,
)
from speechbrain.lobes.models.block_models.utilities.relational_memory import (
    RelationalMemory,
)
from speechbrain.lobes.models.block_models.utilities.RuleNetwork import (
    RuleNetwork,
)

"""
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)
    output:
    output, hx, cx
"""
from torch.distributions.categorical import Categorical


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 1.0

    def backward(ctx, grad_output):
        print(grad_output)
        return grad_output * 1.0


class BlocksCore(nn.Module):
    def __init__(
        self,
        ninp,
        nhid,
        num_blocks_in,
        num_blocks_out,
        topkval,
        memorytopk,
        step_att,
        num_modules_read_input,
        inp_heads,
        do_gru,
        do_rel,
        n_templates,
        share_inp,
        share_comm,
        perm_inv=True,
        memory_slots=4,
        num_memory_heads=4,
        memory_head_size=16,
        memory_mlp=4,
        attention_out=340,
        version=0,
        device=None,
        num_rules=None,
        rule_time_steps=None,
        application_option=3,
        rule_selection="gumble",
    ):
        super(BlocksCore, self).__init__()
        self.nhid = nhid
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.block_size_in = nhid // num_blocks_in
        self.block_size_out = nhid // num_blocks_out
        self.topkval = topkval
        self.memorytopk = memorytopk
        self.step_att = step_att
        self.do_gru = do_gru
        self.do_rel = do_rel
        self.device = device
        self.num_modules_read_input = num_modules_read_input
        self.inp_heads = inp_heads
        self.ninp = ninp
        self.set_transformer = perm_inv
        self.n_templates = n_templates

        print("topk is", self.topkval)
        print("bs in", self.block_size_in)
        print("bs out", self.block_size_out)
        print("inp_heads is", self.inp_heads)
        print("num_modules_read_input", self.num_modules_read_input)
        print("share inp and comm", share_inp, share_comm)
        print("communication is happening", self.step_att)
        print("n_templates:" + str(n_templates))
        print("using set transformer", self.set_transformer)

        if n_templates == 0:
            self.set_transformer = False

        if self.set_transformer:
            self.set = nn.Sequential(
                nn.Linear(self.block_size_out, self.block_size_out),
                nn.ReLU(),
                nn.Linear(self.block_size_out, self.block_size_out),
            )

        self.mha = MultiHeadAttention(
            n_head=4,
            d_model_read=self.block_size_out,
            d_model_write=self.block_size_out,
            d_model_out=self.block_size_out,
            d_k=32,
            d_v=32,
            num_blocks_read=self.num_blocks_out,
            num_blocks_write=self.num_blocks_out,
            dropout=0.1,
            topk=self.num_blocks_out,
            n_templates=1,
            share_comm=share_comm,
            share_inp=False,
            grad_sparse=False,
        )

        self.version = version
        if self.version:
            self.att_out = self.block_size_out
            print("Using version 1 att_out is", self.att_out)
            self.inp_att = MultiHeadAttention(
                n_head=1,
                d_model_read=self.block_size_out,
                d_model_write=int(ninp / self.num_blocks_out),
                d_model_out=self.att_out,
                d_k=64,
                d_v=self.att_out,
                num_blocks_read=1,
                num_blocks_write=num_blocks_in + 1,
                residual=False,
                topk=self.num_blocks_in + 1,
                n_templates=1,
                share_comm=False,
                share_inp=share_inp,
                grad_sparse=False,
                skip_write=True,
            )

        else:
            self.att_out = attention_out
            print("Using version 0 att_out is", self.att_out)
            d_v = self.att_out // self.inp_heads
            self.inp_att = MultiHeadAttention(
                n_head=self.inp_heads,
                d_model_read=self.block_size_out,
                d_model_write=self.block_size_in,
                d_model_out=self.att_out,
                d_k=64,
                d_v=d_v,
                num_blocks_read=num_blocks_out,
                num_blocks_write=self.num_modules_read_input,
                residual=False,
                dropout=0.1,
                topk=self.num_blocks_in + 1,
                n_templates=1,
                share_comm=False,
                share_inp=share_inp,
                grad_sparse=False,
                skip_write=True,
            )

        design_config = {
            "comm": True,
            "grad": False,
            "transformer": True if application_option != 5 else False,
            "application_option": application_option,
            "selection": rule_selection,
        }

        rule_config = {
            "rule_time_steps": rule_time_steps,
            "num_rules": num_rules,
            "rule_emb_dim": 64,
            "rule_query_dim": 32,
            "rule_value_dim": 64,
            "rule_key_dim": 32,
            "rule_heads": 4,
            "rule_dropout": 0.5,
        }

        self.use_rules = (
            rule_config is not None and rule_config["num_rules"] > 0
        )

        if rule_config is not None and rule_config["num_rules"] > 0:
            print("Num Rules:" + str(rule_config["num_rules"]))
            print("Rule Time Steps:" + str(rule_config["rule_time_steps"]))
            self.rule_time_steps = rule_config["rule_time_steps"]
            self.rule_network = RuleNetwork(
                self.block_size_out,
                num_blocks_out,
                num_rules=rule_config["num_rules"],
                rule_dim=rule_config["rule_emb_dim"],
                query_dim=rule_config["rule_query_dim"],
                value_dim=rule_config["rule_value_dim"],
                key_dim=rule_config["rule_key_dim"],
                num_heads=rule_config["rule_heads"],
                dropout=rule_config["rule_dropout"],
                design_config=design_config,
            ).to(device)

        if do_gru:
            print("USING GRU!")
            if n_templates != 0:
                self.block_lstm = SharedBlockGRU(
                    self.att_out * self.num_blocks_out,
                    self.nhid,
                    k=self.num_blocks_out,
                    n_templates=n_templates,
                )
            else:
                print("Using Normal RIMs")
                self.block_lstm = BlockGRU(
                    self.att_out * self.num_blocks_out,
                    self.nhid,
                    k=self.num_blocks_out,
                )
        else:
            print("USING LSTM!")
            if n_templates != 0:
                self.block_lstm = SharedBlockLSTM(
                    self.att_out * self.num_blocks_out,
                    self.nhid,
                    k=self.num_blocks_out,
                    n_templates=n_templates,
                )
            else:
                print("Using Normal RIMs")
                self.block_lstm = BlockLSTM(
                    self.att_out * self.num_blocks_out,
                    self.nhid,
                    k=self.num_blocks_out,
                )

        if self.do_rel:
            memory_key_size = 32
            gate_style = "unit"
            print(
                "gate_style is",
                gate_style,
                memory_slots,
                num_memory_heads,
                memory_head_size,
                memory_key_size,
                memory_mlp,
            )
            self.relational_memory = RelationalMemory(
                mem_slots=memory_slots,
                head_size=memory_head_size,
                input_size=self.nhid,
                output_size=self.nhid,
                num_heads=num_memory_heads,
                num_blocks=1,
                forget_bias=1,
                input_bias=0,
                gate_style="unit",
                attention_mlp_layers=memory_mlp,
                key_size=memory_key_size,
                return_all_outputs=False,
            )

            self.memory_size = memory_head_size * num_memory_heads
            self.mem_att = MultiHeadAttention(
                n_head=4,
                d_model_read=self.block_size_out,
                d_model_write=self.memory_size,
                d_model_out=self.block_size_out,
                d_k=32,
                d_v=32,
                num_blocks_read=self.num_blocks_out,
                num_blocks_write=memory_slots,
                topk=self.num_blocks_out,
                grad_sparse=False,
                n_templates=n_templates,
                share_comm=share_comm,
                share_inp=share_inp,
            )

        self.memory = None

    def blockify_params(self):
        self.block_lstm.blockify_params()

    def forward(
        self, inp, hx, cx, step, do_print=False, message_to_rule_network=None
    ):

        hxl = []
        cxl = []
        sz_b = inp.shape[0]
        batch_size = inp.shape[0]

        inp_use = inp  # layer_input[idx_step]

        def _process_input(_input):
            _input = _input.unsqueeze(1)

            return torch.cat(
                [_input, torch.zeros_like(_input[:, 0:1, :])], dim=1
            )

        if self.version:
            input_to_attention = [
                _process_input(_input)
                for _input in torch.chunk(
                    inp_use, chunks=self.num_blocks_out, dim=1
                )
            ]

            split_hx = [
                chunk.unsqueeze(1)
                for chunk in torch.chunk(hx, chunks=self.num_blocks_out, dim=1)
            ]

            output = [
                self.inp_att(q=_hx, k=_inp, v=_inp)
                for _hx, _inp in zip(split_hx, input_to_attention)
            ]

            inp_use_list, iatt_list, _ = zip(*output)

            inp_use = torch.cat(inp_use_list, dim=1)
            iatt = torch.cat(iatt_list, dim=1)

            inp_use = inp_use.reshape(
                (inp_use.shape[0], self.att_out * self.num_blocks_out)
            )

        else:
            # use attention here.
            inp_use = inp_use.reshape(
                (inp_use.shape[0], self.num_blocks_in, self.block_size_in)
            )
            inp_use = inp_use.repeat(1, self.num_modules_read_input - 1, 1)
            inp_use = torch.cat(
                [torch.zeros_like(inp_use[:, 0:1, :]), inp_use], dim=1
            )
            batch_size = inp.shape[0]
            inp_use, iatt, _ = self.inp_att(
                hx.reshape(
                    (hx.shape[0], self.num_blocks_out, self.block_size_out)
                ),
                inp_use,
                inp_use,
            )
            iatt = iatt.reshape(
                (self.inp_heads, sz_b, iatt.shape[1], iatt.shape[2])
            )
            iatt = iatt.mean(0)

            inp_use = inp_use.reshape(
                (inp_use.shape[0], self.att_out * self.num_blocks_out)
            )

        new_mask = torch.ones_like(iatt[:, :, 0])

        if (self.num_blocks_out - self.topkval) > 0:
            bottomk_indices = torch.topk(
                iatt[:, :, 0],
                dim=1,
                sorted=True,
                largest=True,
                k=self.num_blocks_out - self.topkval,
            )[1]

            new_mask.index_put_(
                (
                    torch.arange(bottomk_indices.size(0)).unsqueeze(1),
                    bottomk_indices,
                ),
                torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype),
            )
        mask = new_mask
        memory_inp_mask = mask
        block_mask = mask.reshape((inp_use.shape[0], self.num_blocks_out, 1))
        mask = (
            mask.reshape((inp_use.shape[0], self.num_blocks_out, 1))
            .repeat((1, 1, self.block_size_out))
            .reshape(
                (inp_use.shape[0], self.num_blocks_out * self.block_size_out)
            )
        )
        mask = mask.detach()
        memory_inp_mask = memory_inp_mask.detach()

        if self.do_gru:
            hx_new, temp_attention = self.block_lstm(inp_use, hx)  # [0]
            cx_new = hx_new
        else:
            hx_new, cx_new, temp_attention = self.block_lstm(inp_use, hx, cx)
        # print(len(hx_new))
        hx_old = hx * 1.0
        cx_old = cx * 1.0

        if self.step_att:  # not self.use_rules:
            hx_new = hx_new.reshape(
                (hx_new.shape[0], self.num_blocks_out, self.block_size_out)
            )
            hx_new_grad_mask = blocked_grad.apply(
                hx_new,
                mask.reshape(
                    (mask.shape[0], self.num_blocks_out, self.block_size_out)
                ),
            )
            hx_new_att, attn_out, extra_loss_att = self.mha(
                hx_new_grad_mask, hx_new_grad_mask, hx_new_grad_mask
            )
            hx_new = hx_new + hx_new_att

            hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
            extra_loss = extra_loss_att

        if self.use_rules:
            hx_new = hx_new.reshape(
                (hx_new.shape[0], self.num_blocks_out, self.block_size_out)
            )
            # encoder_input = encoder_input.reshape((hx_new.shape[0], self.num_blocks_out, self.block_size_out))
            for r in range(self.rule_time_steps):
                hx_new = (
                    self.rule_network(
                        hx_new, message_to_rule_network=message_to_rule_network
                    )
                    + hx_new
                )
            hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))

        hx = (mask) * hx_new + (1 - mask) * hx_old
        cx = (mask) * cx_new + (1 - mask) * cx_old

        if self.do_rel:
            # memory_inp_mask = new_mask
            batch_size = inp.shape[0]
            memory_inp = hx.view(
                batch_size, self.num_blocks_out, -1
            ) * memory_inp_mask.unsqueeze(2)

            # information gets written to memory modulated by the input.
            _, _, self.memory = self.relational_memory(
                inputs=memory_inp.view(batch_size, -1).unsqueeze(1),
                memory=self.memory.cuda(),
            )

            # Information gets read from memory, state dependent information reading from blocks.
            old_memory = self.memory
            out_hx_mem_new, out_mem_2, _ = self.mem_att(
                hx.reshape(
                    (hx.shape[0], self.num_blocks_out, self.block_size_out)
                ),
                self.memory,
                self.memory,
            )
            hx = hx + out_hx_mem_new.reshape(
                hx.shape[0], self.num_blocks_out * self.block_size_out
            )

        if self.set_transformer:
            hx = hx.reshape(
                (hx.shape[0], self.num_blocks_out, self.block_size_out)
            ).reshape((hx.shape[0] * self.num_blocks_out, self.block_size_out))
            hx = (
                self.set(hx)
                .reshape((batch_size, self.num_blocks_out, self.block_size_out))
                .reshape(batch_size, self.nhid)
            )

        # hx = Identity().apply(hx)
        return hx, cx, mask, block_mask, temp_attention

    def reset_relational_memory(self, batch_size: int):
        self.memory = self.relational_memory.initial_state(batch_size).to(
            self.device
        )

    def step_attention(self, hx_new, cx_new, mask):
        hx_new = hx_new.reshape(
            (hx_new.shape[0], self.num_blocks_out, self.block_size_out)
        )
        # bg = blocked_grad()
        hx_new_grad_mask = blocked_grad.apply(
            hx_new,
            mask.reshape(
                (mask.shape[0], self.num_blocks_out, self.block_size_out)
            ),
        )
        hx_new_att, attn_out, extra_loss_att = self.mha(
            hx_new_grad_mask, hx_new_grad_mask, hx_new_grad_mask
        )
        hx_new = hx_new + hx_new_att
        hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
        extra_loss = extra_loss_att
        return hx_new, cx_new, extra_loss


"""
if __name__ == "__main__":
    bc = BlocksCore(512, 1, 4, 4)
    inp = torch.randn(10, 512)
    hx = torch.randn(10,512)
    cx = torch.randn(10,512)
    hx, cx = bc(inp, hx, cx)
    print('hx cx shape', hx.shape, cx.shape)
"""
