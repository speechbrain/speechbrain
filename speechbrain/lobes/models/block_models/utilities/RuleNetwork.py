import torch
import torch.nn as nn
import math
import numpy as np
from .GroupLinearLayer import GroupLinearLayer
from .attention_rim import MultiHeadAttention
import itertools
from .attention import SelectAttention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 1.0

    def backward(ctx, grad_output):
        print(torch.sqrt(torch.sum(torch.pow(grad_output, 2))))
        return grad_output * 1.0


class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        idx = torch.argmax(input, 1)
        ctx._input_shape = input.shape
        ctx._input_dtype = input.dtype
        ctx._input_device = input.device
        # ctx.save_for_backward(idx)
        op = torch.zeros(input.size()).to(input.device)
        op.scatter_(1, idx[:, None], 1)
        ctx.save_for_backward(op)
        return op

    @staticmethod
    def backward(ctx, grad_output):
        (op,) = ctx.saved_tensors
        grad_input = grad_output * op
        return grad_input


class GroupMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num):
        super().__init__()
        self.group_mlp1 = GroupLinearLayer(in_dim, 128, num)
        self.group_mlp2 = GroupLinearLayer(128, out_dim, num)
        # self.group_mlp3 = GroupLinearLayer(128, 128, num)
        # self.group_mlp4 = GroupLinearLayer(128, out_dim, num)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.group_mlp1(x))
        x = self.group_mlp2(x)
        # x = torch.relu(self.dropout(self.group_mlp3(x)))
        # x = torch.relu(self.dropout(self.group_mlp4(x)))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, 128)
        self.mlp2 = nn.Linear(128, out_dim)
        self.mlp3 = nn.Linear(128, 128)
        self.mlp4 = nn.Linear(128, out_dim)
        # self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = self.mlp2(x)
        # x = torch.relu(self.mlp3(x))
        # x = self.mlp4(x)
        # x = torch.relu(self.mlp3(x))
        # x = self.mlp4(x)
        return x


class RuleNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_variables,
        num_rules=4,
        rule_dim=64,
        query_dim=32,
        value_dim=64,
        key_dim=32,
        num_heads=4,
        dropout=0.1,
        design_config=None,
    ):
        super().__init__()
        self.rule_dim = rule_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.value_dim = value_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.design_config = design_config

        self.rule_activation = []
        self.variable_activation = []
        self.softmax = []
        self.masks = []
        import math

        w = 0.01 * torch.randn(1, num_rules, hidden_dim).to(self.device)

        self.rule_embeddings = nn.Parameter(w)
        self.transform_src = nn.Linear(300, 60)

        self.dropout = nn.Dropout(p=0.5)

        self.positional_encoding = PositionalEncoding(hidden_dim)

        self.transform_rule = nn.Linear(rule_dim, hidden_dim)
        if hidden_dim % 4 != 0:
            num_heads = 2
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=0.5
        )

        self.transformer = nn.TransformerEncoder(self.transformer_layer, 3)

        self.multihead_attention = nn.MultiheadAttention(hidden_dim, 4)

        self.variable_rule_select = MultiHeadAttention(
            n_head=4,
            d_model_read=hidden_dim,
            d_model_write=rule_dim,
            d_model_out=hidden_dim,
            d_k=64,
            d_v=64,
            num_blocks_read=2 * num_variables,
            num_blocks_write=num_rules,
            topk=num_rules,
            grad_sparse=False,
        )

        self.encoder_transform = nn.Linear(
            num_variables * hidden_dim, hidden_dim
        )
        self.rule_mlp = GroupMLP(2 * hidden_dim, hidden_dim, num_rules)
        self.rule_linear = GroupLinearLayer(
            rule_dim + hidden_dim, hidden_dim, num_rules
        )
        self.variables_select = MultiHeadAttention(
            n_head=4,
            d_model_read=hidden_dim,
            d_model_write=hidden_dim,
            d_model_out=hidden_dim,
            d_k=64,
            d_v=64,
            num_blocks_read=1,
            num_blocks_write=num_variables,
            topk=3,
            grad_sparse=False,
        )

        self.phase_1_mha = MultiHeadAttention(
            n_head=1,
            d_model_read=2 * hidden_dim * num_variables,
            d_model_write=hidden_dim,
            d_model_out=hidden_dim,
            d_k=64,
            d_v=64,
            num_blocks_read=1,
            num_blocks_write=num_rules,
            topk=num_rules,
            grad_sparse=False,
        )

        self.variable_mlp = MLP(2 * hidden_dim, hidden_dim)
        num = [i for i in range(num_variables)]
        num_comb = len(list(itertools.combinations(num, r=2)))
        self.phase_2_mha = MultiHeadAttention(
            n_head=1,
            d_model_read=hidden_dim,
            d_model_write=hidden_dim,
            d_model_out=hidden_dim,
            d_k=64,
            d_v=64,
            num_blocks_read=num_comb,
            num_blocks_write=1,
            topk=1,
            grad_sparse=False,
        )
        self.variable_mlp_2 = GroupMLP(
            3 * hidden_dim, hidden_dim, num_variables
        )

        # --------Compositonal Search Based Rule Application---------------------------------------
        r = 2
        self.rule_probabilities = []
        self.variable_probabilities = []
        self.r = r
        self.variable_combinations = torch.combinations(
            torch.tensor([i for i in range(num_variables)]),
            r=r,
            with_replacement=True,
        )
        self.variable_combinations_mlp = MLP(r * hidden_dim, hidden_dim)
        self.variable_rule_mlp = MLP(3 * hidden_dim, hidden_dim)
        self.selecter = SelectAttention(
            hidden_dim,
            hidden_dim,
            d_k=16,
            num_read=num_rules,
            num_write=len(self.variable_combinations),
        )
        self.use_rules = MLP(num_variables * hidden_dim, 2)
        self.transform_combinations = MLP(
            len(self.variable_combinations) * hidden_dim, hidden_dim
        )
        self.selecter_1 = SelectAttention(
            hidden_dim, hidden_dim, d_k=16, num_read=1, num_write=num_rules
        )
        self.selecter_2 = SelectAttention(
            hidden_dim,
            hidden_dim,
            d_k=16,
            num_read=1,
            num_write=len(self.variable_combinations),
        )
        self.variable_rule_group_mlp = GroupMLP(
            3 * hidden_dim, hidden_dim, num_rules
        )
        if self.design_config["selection"] == "gumble":
            print("using gumble for rule selection")
        else:
            print("using ArgMax for rule selction")

        print(
            "Using application option "
            + str(self.design_config["application_option"])
        )

        self.gumble_temperature = 1.0

    def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden, message_to_rule_network=None):
        # if not self.design_config['grad']:
        # 	hidden = hidden.detach()

        if False and message_to_rule_network.ndim != hidden.ndim:
            print(
                "dimension of hidden state and message to rule network dont match. Expected both to be 3"
            )
            exit()
        if False and message_to_rule_network.size(2) != hidden.size(2):
            print(
                "Message to rule network should have the same dimension as hidden state. Consider using a linear transform before passing the message."
            )
            exit()
        batch_size, num_variables, variable_dim = hidden.size()

        num_rules = self.rule_embeddings.size(1)
        rule_emb_orig = self.rule_embeddings.repeat(batch_size, 1, 1)
        # print(rule_emb)
        rule_emb = rule_emb_orig

        if False:
            extra_input = message_to_rule_network.detach()
            # extra_input = self.transform_src(extra_input)
            start_index = [0]

            start_index.append(extra_input.size(1))
            start_index.append(start_index[-1] + num_variables)
            # extra_input = self.encoder_transform(extra_input)
            if self.design_config["transformer"]:
                transformer_input = torch.cat(
                    (extra_input, hidden, rule_emb), dim=1
                )
            else:
                read_input = torch.cat((extra_input, hidden), dim=1)
        else:
            start_index = [0, num_variables]
            transformer_input = torch.cat((hidden, rule_emb), dim=1)

        if self.design_config["transformer"]:
            transformer_input = transformer_input.transpose(0, 1)
            transformer_input = self.positional_encoding(transformer_input)
            transformer_out = self.transformer(transformer_input)
            attn_output, attn_output_weights = self.multihead_attention(
                transformer_out, transformer_out, transformer_out
            )
            transformer_out = transformer_out.transpose(0, 1)
            variable_rule = attn_output_weights[
                :,
                start_index[-2] : start_index[-2] + num_variables,
                start_index[-1] :,
            ]
            rule_variable = attn_output_weights[
                :,
                start_index[-1] :,
                start_index[-2] : start_index[-2] + num_variables,
            ].transpose(1, 2)

            scores = variable_rule + rule_variable
            transformer_out = transformer_out.transpose(0, 1)
        else:
            _, variable_rule, _ = self.variable_rule_select(
                read_input, rule_emb_orig, rule_emb_orig
            )
            scores = variable_rule[:, num_variables:, :]
            scores = torch.mean(
                scores.view(
                    rule_emb_orig.size(0), -1, num_variables, num_rules
                ),
                dim=1,
            )
        if self.training:
            mask = torch.nn.functional.gumbel_softmax(
                scores.view(batch_size, -1),
                dim=1,
                tau=self.gumble_temperature,
                hard=True,
            ).view(batch_size, num_variables, num_rules)
            mask_print = (
                ArgMax()
                .apply(scores.view(batch_size, -1))
                .view(batch_size, num_variables, num_rules)
            )
        else:
            mask = (
                ArgMax()
                .apply(scores.view(batch_size, -1))
                .view(batch_size, num_variables, num_rules)
            )
        if self.design_config["application_option"] == 0:
            variable_mask = torch.sum(mask, dim=2).unsqueeze(-1).detach()
            rule_mask = torch.sum(mask, dim=1).unsqueeze(-1).detach()

            # using gumbel for training but printing argmax
            rule_mask_print = (
                torch.sum(mask_print, dim=1).unsqueeze(-1).detach()
            )
            variable_mask_print = (
                torch.sum(mask_print, dim=2).unsqueeze(-1).detach()
            )

            self.rule_activation.append(
                torch.argmax(rule_mask_print, dim=1).detach().cpu().numpy()
            )
            self.variable_activation.append(
                torch.argmax(variable_mask_print, dim=1).detach().cpu().numpy()
            )

            transformer_out = transformer_out.transpose(0, 1)
            scores = scores * mask
            value = transformer_out[:, start_index[-1] :, :]
            rule_mlp_output = torch.matmul(scores, value)
            return rule_mlp_output
        elif self.design_config["application_option"] == 1:
            scores_ = torch.softmax(scores.view(batch_size, -1), dim=1)
            topk = torch.topk(scores_, dim=1, k=2)
            mask = torch.zeros_like(scores_)
            mask.scatter_(1, topk.indices, 1)
            mask = mask.view(batch_size, num_variables, num_rules)
            scores = scores * mask

            rule_mask = torch.sum(scores, dim=1)
            variable_mask = torch.sum(scores, dim=2)
            self.rule_activation.append(topk.indices.cpu().numpy())
            self.variable_activation.append(topk.indices.cpu().numpy())
            transformer_out = transformer_out.transpose(0, 1)
            rules = transformer_out[:, start_index[-1] :, :]
            variables = hidden
            variables = variables.unsqueeze(1)
            variables = variables.repeat(1, rules.size(1), 1, 1)
            variables = variables.view(batch_size, -1, hidden.size(-1))
            rules = rules.repeat(1, hidden.size(1), 1)

            variables = (variables + rules) * scores.view(
                batch_size, -1
            ).unsqueeze(-1)
            variables = variables.view(batch_size, num_variables, num_rules, -1)
            variables = torch.sum(variables, dim=2)

            return variables
        elif self.design_config["application_option"] == 2:
            variable_mask = torch.sum(mask, dim=2).unsqueeze(-1)
            rule_mask = torch.sum(mask, dim=1).unsqueeze(-1)

            selected_variable = (
                torch.sum(hidden * variable_mask, dim=1)
                .unsqueeze(1)
                .repeat(1, mask.size(2), 1)
            )
            rule_mlp_input = torch.cat(
                (rule_emb_orig, selected_variable), dim=2
            )
            rule_mlp_output = self.rule_linear(rule_mlp_input)
            rule_mlp_output = torch.sum(
                rule_mlp_output * rule_mask, dim=1
            ).unsqueeze(1)

            relevant_variables, _, _ = self.variables_select(
                rule_mlp_output, hidden, hidden
            )
            rule_mlp_output = rule_mlp_output + relevant_variables
            rule_mlp_output = rule_mlp_output.repeat(1, hidden.size(1), 1)
            rule_mlp_output = rule_mlp_output * variable_mask
            return rule_mlp_output
        elif self.design_config["application_option"] == 3:
            variable_mask = torch.sum(mask, dim=2).unsqueeze(-1)
            rule_mask = torch.sum(mask, dim=1).unsqueeze(-1)

            # using gumbel for training but printing argmax
            rule_mask_print = (
                torch.sum(mask_print, dim=1).unsqueeze(-1).detach()
            )
            variable_mask_print = (
                torch.sum(mask_print, dim=2).unsqueeze(-1).detach()
            )

            self.rule_activation.append(
                torch.argmax(rule_mask_print, dim=1).detach().cpu().numpy()
            )
            self.variable_activation.append(
                torch.argmax(variable_mask_print, dim=1).detach().cpu().numpy()
            )

            selected_variable = (
                torch.sum(hidden * variable_mask, dim=1)
                .unsqueeze(1)
                .repeat(1, mask.size(2), 1)
            )
            rule_mlp_input = torch.cat(
                (rule_emb_orig, selected_variable), dim=2
            )
            rule_mlp_output = self.rule_mlp(rule_mlp_input)
            rule_mlp_output = torch.sum(
                rule_mlp_output * rule_mask, dim=1
            ).unsqueeze(1)

            relevant_variables, _, _ = self.variables_select(
                rule_mlp_output, hidden, hidden
            )
            rule_mlp_output = rule_mlp_output + relevant_variables
            rule_mlp_output = rule_mlp_output.repeat(1, hidden.size(1), 1)
            rule_mlp_output = rule_mlp_output * variable_mask
            return rule_mlp_output
        elif self.design_config["application_option"] == 4:
            transformer_out = transformer_out.transpose(0, 1)
            scores = scores * mask
            value = transformer_out[:, start_index[-1] :, :]
            rule_mlp_output_1 = torch.matmul(scores, value)
            variable_mask = torch.sum(mask, dim=2).unsqueeze(-1)
            rule_mask = torch.sum(mask, dim=1).unsqueeze(-1)

            selected_variable = (
                torch.sum(hidden * variable_mask, dim=1)
                .unsqueeze(1)
                .repeat(1, mask.size(2), 1)
            )
            rule_mlp_input = torch.cat(
                (rule_emb_orig, selected_variable), dim=2
            )
            rule_mlp_output = self.rule_mlp(rule_mlp_input)
            rule_mlp_output = torch.sum(
                rule_mlp_output * rule_mask, dim=1
            ).unsqueeze(1)

            relevant_variables, _, _ = self.variables_select(
                rule_mlp_output, hidden, hidden
            )
            rule_mlp_output = rule_mlp_output + relevant_variables
            rule_mlp_output = rule_mlp_output.repeat(1, hidden.size(1), 1)
            rule_mlp_output = rule_mlp_output * variable_mask
            return rule_mlp_output + rule_mlp_output_1
        elif self.design_config["application_option"] == 5:
            _, phase_1_attn, _ = self.phase_1_mha(
                read_input.view(read_input.size(0), -1).unsqueeze(1),
                rule_emb,
                rule_emb,
            )
            phase_1_attn = phase_1_attn.squeeze(1)

            mask = torch.nn.functional.gumbel_softmax(
                phase_1_attn, dim=1, tau=0.5, hard=True
            )
            rule_ = torch.argmax(mask, dim=1)
            self.rule_activation.append(rule_.cpu().numpy())
            mask = mask.unsqueeze(-1)

            rule_emb = rule_emb * mask
            rule_emb = torch.sum(rule_emb, dim=1)
            variable_indices = torch.arange(0, num_variables).to(
                rule_emb.device
            )

            variable_indices = torch.combinations(variable_indices, r=2)
            hidden_ = hidden.repeat(1, variable_indices.size(0), 1)
            aux_ind = np.arange(0, variable_indices.size(0))
            aux_ind = np.repeat(aux_ind, 2)
            aux_ind = torch.from_numpy(aux_ind * num_variables).to(
                rule_emb.device
            )
            variable_indices_ = variable_indices.view(-1) + aux_ind
            hidden_ = hidden_[:, variable_indices_, :]

            hidden_ = hidden_.view(hidden_.size(0), -1)
            hidden_ = torch.split(hidden_, 2 * variable_dim, dim=1)
            hidden_ = torch.cat(hidden_, dim=0)
            hidden_ = self.variable_mlp(hidden_)
            hidden_ = torch.split(hidden_, batch_size, dim=0)
            hidden_ = torch.stack(hidden_, dim=1)

            _, variable_attn, _ = self.phase_2_mha(
                hidden_, rule_emb.unsqueeze(1), rule_emb.unsqueeze(1)
            )

            variable_attn = variable_attn.squeeze(-1)
            mask_variable = torch.nn.functional.gumbel_softmax(
                variable_attn, dim=1, hard=True, tau=0.5
            ).unsqueeze(-1)

            hidden_ = hidden_ * mask_variable
            hidden_ = torch.sum(hidden_, dim=1)
            mask_variable_argmax = torch.argmax(mask_variable.squeeze(2), dim=1)
            selected_variable_indices = variable_indices[mask_variable_argmax]
            original_variable_mask = torch.zeros(
                hidden.size(0), hidden.size(1)
            ).to(hidden.device)

            original_variable_mask.scatter_(1, selected_variable_indices, 1)
            original_variable_mask = original_variable_mask.unsqueeze(-1)
            hidden_ = hidden_.unsqueeze(1).repeat(1, hidden.size(1), 1)
            rule_emb = rule_emb.unsqueeze(1).repeat(1, hidden.size(1), 1)
            penultimate_representation = torch.cat(
                (hidden, hidden_, rule_emb), dim=2
            )
            final_representation = (
                self.variable_mlp_2(penultimate_representation)
                * original_variable_mask
            )
            return final_representation
        elif (
            self.design_config["application_option"] == 6
        ):  # compositional search, 2 phase with temperature
            variables = hidden  # (B, num_variables, variable_dim)
            variable_combinations = (
                self.variable_combinations
            )  # (num_combinations, r)
            variable_combinations_mask = torch.zeros(
                variable_combinations.size(0), variables.size(1)
            )
            variable_combinations_mask.scatter_(1, variable_combinations, 1)

            # use_rules = self.use_rules(variables.view(variables.size(0), -1))
            # use_rules = torch.nn.functional.gumbel_softmax(use_rules, dim = 1, tau = 0.5, hard = True) # (B, 2) index(0) = 1->no rule index(1) = 1->rule
            # use_rules = torch.split(use_rules, 1, dim = 1)
            # use_rule = use_rules[1].unsqueeze(-1).repeat(1, self.r, 1) # (B, r, 1)

            variables_repeat = variables  # .repeat(1, variable_combinations.size(0), 1) # (B, num_combination * num_variables, variable_dim)
            variables_extract = variables_repeat[
                :, variable_combinations.view(-1), :
            ]  # (B, r * num_combinations, variable_dim)

            variables_extract = list(
                torch.split(variables_extract, self.r, dim=1)
            )  # [(B, r, variable_dim) * num_combinations]

            # Find a way to avoid loop!
            variables_extract = [
                v.view(variables.size(0), -1) for v in variables_extract
            ]  # [(B, r * variable_dim ) * num_combinations]
            variables_extract = torch.stack(
                variables_extract, dim=1
            )  # (B, num_combinations, r * variable_dim)

            combined_variable_representations = self.variable_combinations_mlp(
                variables_extract
            )  # (B, num_combinations, variable_dim)
            # no_op_rule = torch.zeros(rule_emb.size(0), 1, rule_emb.size(2)).to(rule_emb.device)

            rules = rule_emb  # torch.cat((no_op_rule, rule_emb), dim = 1) # (B, num_rules, rule_dim)
            rules_mask = torch.ones(rules.size()).to(rules.device)

            num_combinations = combined_variable_representations.size(1)
            num_rules = rules.size(1)
            # print(self.rule_embeddings[0,:, 5])
            # transformer_input = torch.cat((combined_variable_representations, rules), dim = 1) # (B, num_combinations + num_rules, variable_dim)

            # transformer_input = transformer_input.transpose(0, 1)
            # transformer_input = self.positional_encoding(transformer_input)
            # transformer_out = self.transformer(transformer_input)
            # attn_output, attn_output_weights = self.multihead_attention(transformer_out, transformer_out, transformer_out)
            # transformer_out  = transformer_out.transpose(0, 1)

            # variable_rule = attn_output_weights[:, :num_combinations,  num_combinations:] # (B, num_combinations, num_rules)
            # rule_variable = attn_output_weights[:,  num_combinations:, : num_combinations].transpose(1, 2) # (B, num_rules, num_combinations) -> (B, num_combinations, num_rules)

            # scores = variable_rule + rule_variable
            # mask = torch.nn.functional.gumbel_softmax(scores.view(batch_size, -1), dim = 1, tau = 0.5, hard = True).view(batch_size, num_combinations, num_rules)

            merged_variable_representation = combined_variable_representations.reshape(
                combined_variable_representations.size(0), -1
            )  # (B, num_combinations * variable_dim)
            merged_variable_representation = self.transform_combinations(
                merged_variable_representation
            )

            scores_1 = self.selecter_1(
                merged_variable_representation.unsqueeze(1), rules
            )
            # scores_1 = scores_1.reshape(rules.size(0), -1, scores_1.size(1), scores_1.size(2))
            # scores_1 = torch.mean(scores_1, dim = 1)
            scores_1 = scores_1.squeeze(1)  # scores_1: (B, num_rules)
            self.rule_probabilities.append(
                torch.softmax(scores_1.clone(), dim=1).detach().cpu().numpy()
            )
            if self.design_config["selection"] == "gumble":
                mask_rule = torch.nn.functional.gumbel_softmax(
                    scores_1, dim=1, tau=self.gumble_temperature, hard=True
                )
            else:
                mask_rule = ArgMax().apply(scores_1)

            selected_rule = rules * mask_rule.unsqueeze(-1)
            selected_rule = torch.sum(
                selected_rule, dim=1
            )  # (B, rule_dim)  rule_dim = variable_dim

            scores_2 = self.selecter_2(
                selected_rule.unsqueeze(1), combined_variable_representations
            )

            scores_2 = scores_2.squeeze(1)  # (B, num_combinations)
            self.variable_probabilities.append(
                torch.softmax(scores_2.clone(), dim=1).detach().cpu().numpy()
            )

            if self.design_config["selection"] == "gumble" and self.training:
                mask_variable = torch.nn.functional.gumbel_softmax(
                    scores_2, dim=1, tau=self.gumble_temperature, hard=True
                )
            else:
                mask_variable = ArgMax().apply(scores_2)

            rule_activation = torch.argmax(mask_rule, dim=1).detach()
            self.rule_activation.append(
                rule_activation.detach().int().cpu().numpy()
            )
            self.variable_activation.append(
                variable_combinations[
                    torch.argmax(mask_variable, dim=1).detach()
                ]
                .cpu()
                .numpy()
            )

            combination_selections = mask_variable
            combinations = torch.argmax(combination_selections, dim=1).detach()

            rule = selected_rule.unsqueeze(1)
            combination_selections = combination_selections.unsqueeze(-1)

            combined_variable = torch.bmm(
                mask_variable.unsqueeze(1), combined_variable_representations
            )  # combined_variable_representations * combination_selections # (B, num_combinations, variable_dim)
            # combined_variable = combined_variable.mean(1)
            # print(combined_variable.size())

            # combined_variable = torch.sum(combined_variable, dim = 1).unsqueeze(1) # (B, variable_dim)
            #### gradcheck grad / 10 (same as line 411)
            temp_index = torch.zeros(combined_variable.size(0), self.r).to(
                combined_variable.device
            ) + torch.arange(0, combined_variable.size(0)).to(
                combined_variable.device
            ).unsqueeze(
                -1
            )

            combinations = variable_combinations[combinations]

            selected_variables = variables[
                temp_index.long(), combinations.long()
            ]
            # selected_variables.requires_grad = True

            combined_variable = combined_variable.repeat(1, self.r, 1)
            rule = rule.repeat(1, self.r, 1)
            variable_rule_representations = torch.cat(
                (selected_variables, combined_variable, rule), dim=2
            )
            penultimate_representation = (
                self.variable_rule_mlp(variable_rule_representations)
                + combined_variable
                + rule
            )
            ####gradcheck grad / 10

            mask_index = np.arange(rules_mask.size(0))

            variable_rule = penultimate_representation
            residual_application_mask = torch.zeros(variables.size()).to(
                variable_rule.device
            )
            residual_application_mask[
                temp_index.long(), combinations.long()
            ] = (
                residual_application_mask[
                    temp_index.long(), combinations.long()
                ]
                + variable_rule
            )

            return residual_application_mask
        elif self.design_config["application_option"] == 7:
            variables = hidden  # (B, num_variables, variable_dim)
            variable_combinations = (
                self.variable_combinations
            )  # (num_combinations, r)
            variable_combinations_mask = torch.zeros(
                variable_combinations.size(0), variables.size(1)
            )
            variable_combinations_mask.scatter_(1, variable_combinations, 1)

            variables_repeat = variables  # .repeat(1, variable_combinations.size(0), 1) # (B, num_combination * num_variables, variable_dim)
            variables_extract = variables_repeat[
                :, variable_combinations.view(-1), :
            ]  # (B, r * num_combinations, variable_dim)

            variables_extract = list(
                torch.split(variables_extract, self.r, dim=1)
            )  # [(B, r, variable_dim) * num_combinations]

            # Find a way to avoid loop!
            variables_extract = [
                v.view(variables.size(0), -1) for v in variables_extract
            ]  # [(B, r * variable_dim ) * num_combinations]
            variables_extract = torch.stack(
                variables_extract, dim=1
            )  # (B, num_combinations, r * variable_dim)

            combined_variable_representations = self.variable_combinations_mlp(
                variables_extract
            )  # (B, num_combinations, variable_dim)

            rules = rule_emb  # torch.cat((no_op_rule, rule_emb), dim = 1) # (B, num_rules, rule_dim)
            rules_mask = torch.ones(rules.size()).to(rules.device)

            num_combinations = combined_variable_representations.size(1)
            num_rules = rules.size(1)

            scores = self.selecter(
                rules.detach(), combined_variable_representations.detach()
            )

            if self.design_config["selection"] == "gumble" and self.training:
                mask = torch.nn.functional.gumbel_softmax(
                    scores.view(rules.size(0), -1),
                    dim=1,
                    tau=self.gumble_temperature,
                    hard=True,
                ).view(-1, num_rules, num_combinations)
            else:
                mask = (
                    ArgMax()
                    .apply(scores.view(rules.size(0), -1))
                    .view(-1, num_rules, num_combinations)
                )

            mask = mask.permute(0, 2, 1)
            mask_rule = torch.sum(mask, dim=1)
            mask_variable = torch.sum(mask, dim=2)

            selected_rule = rules * mask_rule.unsqueeze(-1)
            selected_rule = torch.sum(selected_rule, dim=1)

            rule_activation = torch.argmax(mask_rule, dim=1).detach()

            self.rule_activation.append(
                rule_activation.detach().int().cpu().numpy()
            )
            self.variable_activation.append(
                variable_combinations[
                    torch.argmax(mask_variable, dim=1).detach()
                ]
                .cpu()
                .numpy()
            )
            self.rule_probabilities.append(
                torch.softmax(
                    scores.clone().detach().view(rules.size(0), -1), dim=1
                )
                .cpu()
                .numpy()
            )
            self.variable_probabilities.append(
                torch.softmax(
                    scores.clone().detach().view(rules.size(0), -1), dim=1
                )
                .cpu()
                .numpy()
            )

            combination_selections = mask_variable
            combinations = torch.argmax(combination_selections, dim=1).detach()

            rule = selected_rule.unsqueeze(1)
            combination_selections = combination_selections.unsqueeze(-1)

            combined_variable = torch.bmm(
                mask_variable.unsqueeze(1), combined_variable_representations
            )  # combined_variable_representations * combination_selections # (B, num_combinations, variable_dim)

            temp_index = torch.zeros(combined_variable.size(0), self.r).to(
                combined_variable.device
            ) + torch.arange(0, combined_variable.size(0)).float().to(
                combined_variable.device
            ).unsqueeze(
                -1
            )

            combinations = variable_combinations[combinations]

            selected_variables = variables[
                temp_index.long(), combinations.long()
            ]
            # selected_variables.requires_grad = True

            combined_variable = combined_variable.repeat(1, self.r, 1)
            rule = rule.repeat(1, self.r, 1)
            variable_rule_representations = torch.cat(
                (selected_variables, combined_variable, rule), dim=2
            )
            penultimate_representation = (
                self.variable_rule_mlp(variable_rule_representations)
                + combined_variable
                + rule
            )

            mask_index = np.arange(rules_mask.size(0))

            variable_rule = penultimate_representation
            residual_application_mask = torch.zeros(variables.size()).to(
                variable_rule.device
            )
            residual_application_mask[
                temp_index.long(), combinations.long()
            ] = (
                residual_application_mask[
                    temp_index.long(), combinations.long()
                ]
                + variable_rule
            )

            return residual_application_mask

        elif self.design_config["application_option"] == 8:
            variables = hidden  # (B, num_variables, variable_dim)
            variable_combinations = (
                self.variable_combinations
            )  # (num_combinations, r)
            variable_combinations_mask = torch.zeros(
                variable_combinations.size(0), variables.size(1)
            )
            variable_combinations_mask.scatter_(1, variable_combinations, 1)

            variables_repeat = variables  # .repeat(1, variable_combinations.size(0), 1) # (B, num_combination * num_variables, variable_dim)
            variables_extract = variables_repeat[
                :, variable_combinations.view(-1), :
            ]  # (B, r * num_combinations, variable_dim)

            variables_extract = list(
                torch.split(variables_extract, self.r, dim=1)
            )  # [(B, r, variable_dim) * num_combinations]

            # Find a way to avoid loop!
            variables_extract = [
                v.view(variables.size(0), -1) for v in variables_extract
            ]  # [(B, r * variable_dim ) * num_combinations]
            variables_extract = torch.stack(
                variables_extract, dim=1
            )  # (B, num_combinations, r * variable_dim)

            combined_variable_representations = self.variable_combinations_mlp(
                variables_extract
            )  # (B, num_combinations, variable_dim)

            rules = rule_emb  # torch.cat((no_op_rule, rule_emb), dim = 1) # (B, num_rules, rule_dim)
            rules_mask = torch.ones(rules.size()).to(rules.device)

            num_combinations = combined_variable_representations.size(1)
            num_rules = rules.size(1)

            scores = self.selecter(rules, combined_variable_representations)
            if self.design_config["selection"] == "gumble" and self.training:
                mask = torch.nn.functional.gumbel_softmax(
                    scores.view(rules.size(0), -1),
                    dim=1,
                    tau=self.gumble_temperature,
                    hard=True,
                ).view(-1, num_rules, num_combinations)
            else:
                mask = (
                    ArgMax()
                    .apply(scores.view(rules.size(0), -1))
                    .view(-1, num_rules, num_combinations)
                )

            mask = mask.permute(0, 2, 1)
            mask_rule = torch.sum(mask, dim=1)
            mask_variable = torch.sum(mask, dim=2)

            selected_rule = rules * mask_rule.unsqueeze(-1)
            selected_rule = torch.sum(selected_rule, dim=1)

            rule_activation = torch.argmax(mask_rule, dim=1).detach()

            self.rule_activation.append(
                rule_activation.detach().int().cpu().numpy()
            )
            self.variable_activation.append(
                variable_combinations[
                    torch.argmax(mask_variable, dim=1).detach()
                ]
                .cpu()
                .numpy()
            )
            self.rule_probabilities.append(
                torch.softmax(
                    scores.clone().detach().view(rules.size(0), -1), dim=1
                )
                .cpu()
                .numpy()
            )
            self.variable_probabilities.append(
                torch.softmax(
                    scores.clone().detach().view(rules.size(0), -1), dim=1
                )
                .cpu()
                .numpy()
            )

            combination_selections = mask_variable
            combinations = torch.argmax(combination_selections, dim=1).detach()

            rule = selected_rule.unsqueeze(1)
            combination_selections = combination_selections.unsqueeze(-1)

            combined_variable = torch.bmm(
                mask_variable.unsqueeze(1), combined_variable_representations
            )  # combined_variable_representations * combination_selections # (B, num_combinations, variable_dim)

            temp_index = torch.zeros(combined_variable.size(0), self.r).to(
                combined_variable.device
            ) + torch.arange(0, combined_variable.size(0)).to(
                combined_variable.device
            ).unsqueeze(
                -1
            )

            combinations = variable_combinations[combinations]

            selected_variables = variables[
                temp_index.long(), combinations.long()
            ]
            selected_variables.requires_grad = True

            combined_variable = combined_variable.repeat(1, self.r, 1)
            rule = rule.repeat(1, self.r, 1)
            variable_rule_representations = torch.cat(
                (selected_variables, combined_variable, rule), dim=2
            )  # (B, 2, 3 * variable_dim)
            variable_rule_representations = variable_rule_representations.view(
                variable_rule_representations.size(0) * self.r, -1
            ).unsqueeze(1)
            variable_rule_representations = variable_rule_representations.repeat(
                1, num_rules, 1
            )
            variable_rule_representations = self.variable_rule_group_mlp(
                variable_rule_representations
            )

            variable_rule_representations = variable_rule_representations.reshape(
                rule.size(0), self.r, num_rules, -1
            )
            variable_rule_representations = variable_rule_representations * mask_rule.unsqueeze(
                1
            ).unsqueeze(
                -1
            )
            variable_rule_representations = torch.sum(
                variable_rule_representations, dim=2
            )

            penultimate_representation = (
                variable_rule_representations + combined_variable + rule
            )

            mask_index = np.arange(rules_mask.size(0))

            variable_rule = penultimate_representation
            residual_application_mask = torch.zeros(variables.size()).to(
                variable_rule.device
            )
            residual_application_mask[
                temp_index.long(), combinations.long()
            ] = (
                residual_application_mask[
                    temp_index.long(), combinations.long()
                ]
                + variable_rule
            )

            return residual_application_mask

    def reset_activations(self):
        self.rule_activation = []
        self.variable_activation = []
        self.rule_probabilities = []
        self.variable_probabilities = []


if __name__ == "__main__":
    model = RuleNetwork(6, 4).cuda()

    hiddens = torch.autograd.Variable(
        torch.randn(3, 4, 6), requires_grad=True
    ).cuda()
    new_hiddens = model(hiddens)

    hiddens.retain_grad()
    new_hiddens.backward(torch.ones(hiddens.size()).cuda())

    # print(model.rule_embeddings.grad)
    # print(model.query_layer.w.grad)
