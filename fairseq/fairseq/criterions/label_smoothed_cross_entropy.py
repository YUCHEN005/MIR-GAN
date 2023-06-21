# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    weight_gan: float = field(
        default=0.01,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    weight_mim: float = field(
        default=0.005,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    logit_temp: float = field(
        default=0.1,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        weight_gan=0.01,
        weight_mim=0.005,
        logit_temp=0.1,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.weight_gan = weight_gan
        self.weight_mim = weight_mim
        self.logit_temp = logit_temp
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, feats = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        Discriminator = model.encoder.w2v_model.modal_discriminator
        x_a, x_v, x = feats
        B, T, D = x.shape

        ### discriminator loss
        loss_d = torch.log(Discriminator(x_a.clone().detach())) + torch.log(1 - Discriminator(x_v.clone().detach()))
        mask_d = torch.isfinite(loss_d) & ~torch.isnan(loss_d)
        loss_d = loss_d.masked_select(mask_d).sum() / B

        loss_g = - torch.log(Discriminator(x.clone().detach())) - torch.log(1 - Discriminator(x.clone().detach()))
        mask_g = torch.isfinite(loss_g) & ~torch.isnan(loss_g)
        loss_g = loss_g.masked_select(mask_g).sum() / B

        loss_gan = loss_d + loss_g

        ### generator loss
        loss_G = - torch.log(Discriminator(x)) - torch.log(1 - Discriminator(x))
        mask_G = torch.isfinite(loss_G) & ~torch.isnan(loss_G)
        loss_G = loss_G.masked_select(mask_G).sum() / B

        ### mim loss
        assert x_a.shape == x_v.shape == x.shape, (x_a.shape, x_v.shape, x.shape)
        logits_a = torch.matmul(x, x_a.transpose(1, 2)) / self.logit_temp     # (B, T, T)
        targets_a = torch.arange(T).unsqueeze(0).repeat(B, 1).to(logits_a.device)
        loss_a = self.ce_loss(logits_a, targets_a)
        mask_a = torch.isfinite(loss_a) & ~torch.isnan(loss_a)
        loss_a = loss_a.masked_select(mask_a).mean()

        logits_v = torch.matmul(x, x_v.transpose(1, 2)) / self.logit_temp     # (B, T, T)
        targets_v = torch.arange(T).unsqueeze(0).repeat(B, 1).to(logits_v.device)
        loss_v = self.ce_loss(logits_v, targets_v)
        mask_v = torch.isfinite(loss_v) & ~torch.isnan(loss_v)
        loss_v = loss_v.masked_select(mask_v).mean()

        loss_mim = loss_a + loss_v

        ### final loss
        loss = loss + loss_G * self.weight_gan + loss_mim * self.weight_mim

        # log output
        logging_output["loss"] = loss.data
        logging_output["loss_D"] = loss_d.data
        logging_output["loss_G"] = loss_g.data
        logging_output["loss_mim"] = loss_mim.data

        return - loss_gan * self.weight_gan, loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_D_sum = sum(log.get("loss_D", 0) for log in logging_outputs)
        loss_G_sum = sum(log.get("loss_G", 0) for log in logging_outputs)
        loss_mim_sum = sum(log.get("loss_mim", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_D", loss_D_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_G", loss_G_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_mim", loss_mim_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
