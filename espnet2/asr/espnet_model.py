import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

# OSWALD
import numpy
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: Optional[dict] = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        blocks_training: int = 0,
        random_blocks: int = 0,
        uniform_sampling: bool = False,
        is_self_distilling: bool = False,
        distill_weight: float = 0.0,
        use_timing_loss: bool = False,
        use_single_head: bool = False,
        only_last_timing: bool = False,
        only_last_layer_timing: bool = True,
        timing_loss_weight: float = 0.0,
        use_last_head_distill: bool = False,
        use_last_layer_distill: bool = False,
        regression_timing_weight: float = 0.,
        use_regression_timing: bool = False,
        use_tuple_loss: bool = False,
        tuple_loss_weight: float = 0.,
        output_dir: str = "",
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        #                  which doesn't use <blank> token
        if sym_blank in token_list:
            self.blank_id = token_list.index(sym_blank)
        else:
            self.blank_id = 0
        if sym_sos in token_list:
            self.sos = token_list.index(sym_sos)
        else:
            self.sos = vocab_size - 1
        if sym_eos in token_list:
            self.eos = token_list.index(sym_eos)
        else:
            self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.aux_ctc = aux_ctc
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder


        # OSWALD:
        self.blocks_training = blocks_training
        logging.info(f"[Approach 2]: {self.blocks_training=}")
        self.random_blocks = random_blocks
        logging.info(f"{self.random_blocks}")
        self.uniform_sampling=uniform_sampling
        self.is_self_distilling = is_self_distilling
        logging.info(f"[Approach 2]: {self.is_self_distilling}")
        self.distill_weight = distill_weight
        logging.info(f"[Approach 2]: {self.distill_weight=}")
        self.use_timing_loss = use_timing_loss
        self.use_single_head = use_single_head
        self.only_last_timing = only_last_timing
        self.only_last_layer_timing = only_last_layer_timing
        self.timing_loss_weight = timing_loss_weight
        self.use_last_head_distill = use_last_head_distill
        self.use_last_layer_distill = use_last_layer_distill
        self.use_tuple_loss = use_tuple_loss
        self.tuple_loss_weight = tuple_loss_weight
        self.saved_examples = {}
        self.output_dir = output_dir


        if not hasattr(self.encoder, "interctc_use_conditioning"):
            self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            self.decoder = decoder
            self.joint_network = joint_network

            if not transducer_multi_blank_durations:
                from warprnnt_pytorch import RNNTLoss

                self.criterion_transducer = RNNTLoss(
                    blank=self.blank_id,
                    fastemit_lambda=0.0,
                )
            else:
                from espnet2.asr.transducer.rnnt_multi_blank.rnnt_multi_blank import (
                    MultiblankRNNTLossNumba,
                )

                self.criterion_transducer = MultiblankRNNTLossNumba(
                    blank=self.blank_id,
                    big_blank_durations=transducer_multi_blank_durations,
                    sigma=transducer_multi_blank_sigma,
                    reduction="mean",
                    fastemit_lambda=0.0,
                )
                self.transducer_multi_blank_durations = transducer_multi_blank_durations

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight < 1.0:
                assert (
                    decoder is not None
                ), "decoder should not be None when attention is used"
            else:
                decoder = None
                logging.warning("Set decoder to none as ctc_weight==1.0")

            self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

        if lang_token_id != -1:
            self.lang_token_id = torch.tensor([[lang_token_id]])
        else:
            self.lang_token_id = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        # OSWALD
        epoch: int = -1,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        # OSWALD
        self.epoch = epoch

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        if self.is_self_distilling and self.blocks_training != 0:
            # logging.info(f"[Approach 2] [Self-distillation]: Doing non-masked enc forward pass.")
            encoder_out, encoder_out_no_masking, encoder_out_lens = self.encode(speech, speech_lengths, also_full_context=True, **kwargs)
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, **kwargs)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths)

            # [OSWALD]: Nach attention loss sollte, decoder normale maskierte attn drin haben
            attn_masked_context = []
            for decoder in self.decoder.decoders:
                attn_masked_context.append(decoder.src_attn.attn)
            attn_masked_context = torch.stack(attn_masked_context)

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # [OSWALD]: KLDivLoss expects input to be in log space
            if self.is_self_distilling:
                kl_loss = self._calc_kl_loss(encoder_out_no_masking, encoder_out_lens, text, text_lengths, attn_masked_context, utt_ids = kwargs['utt_id'])
                stats['kl_loss'] = kl_loss * self.distill_weight
                loss = loss + self.distill_weight * kl_loss

            if self.use_timing_loss:
                use_libri_timings = 'w_timing_libri' in kwargs
                if 'w_timing' in kwargs:
                    w_timing = kwargs['w_timing']
                    w_timing_lengths = kwargs['w_timing_lengths']
                elif use_libri_timings:
                    w_timing = kwargs['w_timing_libri']
                    w_timing_lengths = kwargs['w_timing_libri_lengths']
                utt_id = kwargs['utt_id']
                timing_loss = self._calc_timing_loss(attn_masked_context, w_timing, w_timing_lengths, text, text_lengths, utt_id = utt_id, encoder_out_lens=encoder_out_lens, use_libri_timings=use_libri_timings)
                stats['timing_loss'] = timing_loss * self.timing_loss_weight
                loss = loss + self.timing_loss_weight * timing_loss

            if self.use_tuple_loss:
                use_libri_timings = 'w_timing_libri' in kwargs
                if 'w_timing' in kwargs:
                    w_timing = kwargs['w_timing']
                    w_timing_lengths = kwargs['w_timing_lengths']
                elif use_libri_timings:
                    w_timing = kwargs['w_timing_libri']
                    w_timing_lengths = kwargs['w_timing_libri_lengths']
                utt_id = kwargs['utt_id']
                tuple_loss = self._calc_tuple_loss(w_timing, use_libri_timings, utt_id, encoder_out, encoder_out_lens, text, text_lengths)
                stats['tuple_loss'] = tuple_loss * self.tuple_loss_weight
                loss = loss + self.tuple_loss_weight * tuple_loss


            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        # OSWALD
        blocks_inference: int = 0,
        also_full_context: bool = False,
        is_inference: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """

        # if not torch.all(speech_lengths == speech_lengths[0]):
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # [OSWALD]: Always set decoder blocks_inference in decoder
        self.decoder.blocks_inference = blocks_inference

        # Pre-encoder, e.g. used for raw input data
        debug_logging=False

        if also_full_context:
            feats_full_context = feats.clone().detach()

        # Start Masking from t_{EOS} <=> equivalent to adding t_{EOA} - t_{EOU} to masking
        if "timing_path" in kwargs: # use mfa timings (for eval):
            import textgrid
            from pathlib import Path
            utt_id = kwargs['utt_id']
            filepath = Path(kwargs["timing_path"]) / (kwargs["utt_id"][0] + ".TextGrid") # inference always use batchsize 1 -> only one utt_id
            tg = textgrid.TextGrid.fromFile(filepath)
            spans = [x for x in tg[0] if x.mark != '']
            lasto = spans[-1].maxTime
            # OSWALD: in eval there is no speed perturb so no scaling needed
            lasto = lasto * 100 # in 10ms blocks
            additional_blocks = feats_lengths - lasto
        else:
            use_libri_timings = 'w_timing_libri' in kwargs
            if not use_libri_timings:
                # iirc in 10ms steps = 1 frame
                utt_id = kwargs['utt_id']
                w_timing = kwargs['w_timing']
                start_times = torch.tensor([int(x.split('_')[1].split('-')[0]) for x in utt_id])
                mask = w_timing != -1
                w_timing = torch.where(mask, w_timing - start_times.unsqueeze(-1).to(w_timing.device), w_timing)
                w_timing = w_timing.float()
                for i in range(len(utt_id)):
                    if 'sp0.9-' in utt_id[i]:
                        w_timing[i][w_timing[i] != -1] /= 0.9
                    elif 'sp1.1-' in utt_id[i]:
                        w_timing[i][w_timing[i] != -1] /= 1.1
                w_timing = torch.round(w_timing).int()
                eous = w_timing.gather(1, (mask.sum(dim=1) - 1).unsqueeze(-1))
                additional_blocks = feats_lengths - eous.squeeze()
                if additional_blocks.min() < -1:
                    print(additional_blocks, w_timing, feats_lengths)
                    print(utt_id)
                    import pdb;pdb.set_trace()

        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # Mask training blocks
        if self.blocks_training != 0 and not is_inference:
            if debug_logging:
                logging.info(f"[Approach 2]: Masking encoder frames: {self.blocks_training=}")
            if not self.uniform_sampling:
                exit() # gaussian sucks
                mask_sub_val = numpy.abs(numpy.random.normal(loc = 0.0, scale = self.blocks_training, size=feats_lengths.shape))
            else:
                np_rng = numpy.random.default_rng()
                mask_sub_val = np_rng.uniform(0, self.blocks_training, size=feats_lengths.shape)
            mask_sub_val = torch.tensor(mask_sub_val).int().to(feats_lengths.device)

            self.current_masking_blocks = mask_sub_val.detach().cpu().numpy()
            # Save the masking amount for plotting purposes in decoder
            self.decoder.mask_sub_val = mask_sub_val

            total_blocks = mask_sub_val + additional_blocks

            for i in range(feats.shape[0]): # loop over batch
                # Mask feats to 0.0
                feats[i, feats_lengths[i] - total_blocks[i] : feats_lengths[i], :] = 0.0
        # Mask inference blocks
        if blocks_inference != 0 and is_inference:
            test_set_decoding = True
            if not test_set_decoding:
                exit() # Prolly not what I want during my thesis
                if debug_logging:
                    logging.info(f"[Approach 2]: Adding {blocks_inference} inference blocks. This mode intended for live mode with full audio. Increase feats_lengths from {feats_lengths} to {feats_lengths + blocks_inference}")
                tgt_shape = list(feats.shape)
                tgt_shape[1] = blocks_inference
                feats = torch.cat([feats, torch.zeros(tgt_shape).to(feats.device)], 1)
                feats_lengths = feats_lengths + blocks_inference
            else:
                # TODO change to mask from t_{EOS}
                total_inf_blocks = additional_blocks + blocks_inference
                total_inf_blocks = int(total_inf_blocks.item())
                feats[:, -total_inf_blocks:, :] = 0.0
                if debug_logging:
                    logging.info(f"[Approach 2]: Zeroing out the last {blocks_inference} blocks. For live inference thecode needs to be adjusted.")
                    logging.info(f"[Approach 2 DEBUG]: Total encoder feats_lengths {feats_lengths}. This corresponds to {feats_lengths * 10} ms audio len .")

        # RANDOM BLOCKS
        if self.random_blocks != 0 and not is_inference:
            if debug_logging:
                logging.info(f"[Approach 2]: Using {self.random_blocks} random_blocks. Adding random amount of pos. encoding blocks with mean {self.random_blocks}. This message should not appear during inference.")
            if not self.uniform_sampling:
                random_block_val = numpy.random.normal(loc = 0.0, scale = self.random_blocks, size=feats_lengths.shape)
                exit() # gaussian = bad
            else:
                np_rng = numpy.random.default_rng()
                random_block_val = np_rng.uniform(0, self.random_blocks, size=feats_lengths.shape)
            random_block_val = torch.tensor(random_block_val).int().to(feats_lengths.device)
            # Save random block val to decoder for plotting
            self.decoder.random_block_val = random_block_val
            self.current_random_blocks = random_block_val.detach().cpu().numpy()

            # Used for new padding
            cur_len = feats.shape[1]
            max_new_len = (random_block_val + feats_lengths).max()
            pad_needed = max(0, max_new_len - cur_len)

            tgt_shape = list(feats.shape)
            tgt_shape[1] = pad_needed

            feats = torch.cat([feats, torch.zeros(tgt_shape).to(feats.device)], 1)
            feats_lengths = feats_lengths + random_block_val.type(feats_lengths.dtype)
            feats_lengths = torch.clamp(feats_lengths, min=0)

            if also_full_context:
                feats_full_context = torch.cat([feats_full_context, torch.zeros(tgt_shape).to(feats.device)], 1)

            if debug_logging:
                logging.info(f"[Approach 2]:after: {feats.shape=}")
                logging.info(f"[Approach 2]:after: {feats_lengths}")


        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        # OSWALD nope
        if self.encoder.interctc_use_conditioning or getattr(
            self.encoder, "ctc_trim", False
        ):
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
            if also_full_context:
                # Yosuke: eval mode and no grad
                with torch.no_grad():
                    self.encoder.eval()
                    encoder_out_full_context, enc_out_lens_fc, _ = self.encoder(feats_full_context, feats_lengths)
                    self.encoder.train()
                    assert torch.equal(enc_out_lens_fc, encoder_out_lens)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # Post-encoder, e.g. NLU
        # OSWALD: Nope
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
            and not self.random_blocks != 0
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        if also_full_context:
            assert encoder_out.shape == encoder_out_full_context.shape
            return encoder_out, encoder_out_full_context, encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_kl_loss(self, encoder_out_unmasked, encoder_out_lens, ys_pad, ys_pad_lens, masked_x_attn, utt_ids = None):
        # 1. Get cross attention of decoder with unmasked encoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        padding_mask_encoder = make_pad_mask(encoder_out_lens)
        padding_mask_encoder = padding_mask_encoder.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(masked_x_attn.shape)

        # Make mask for decoder
        batch_size = masked_x_attn.shape[1]
        max_value = max(ys_in_lens)
        dec_mask = torch.arange(max_value).unsqueeze(0).expand(batch_size, max_value).to(masked_x_attn.device)
        dec_mask = dec_mask >= ys_in_lens.unsqueeze(1)
        # B x D -> LxBxHxDxE
        dec_mask = dec_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1).expand(masked_x_attn.shape)
        # Masks; where True is set there masking should be applied. ==> combination should be OR
        combined_mask = torch.logical_or(dec_mask, padding_mask_encoder)

        with torch.no_grad():
            self.decoder.eval()
            _, _ = self.decoder(encoder_out_unmasked, encoder_out_lens, ys_in_pad, ys_in_lens)
            self.decoder.train()
        # List [0 - 5] with shape B x H x D x E
        unmasked_attn_full = []
        for decoder in self.decoder.decoders:
            unmasked_attn_full.append(decoder.src_attn.attn[:, :, :, :]) # All heads for plots
        unmasked_attn_full = torch.stack(unmasked_attn_full)

        if self.use_last_layer_distill:
            masked_x_attn = masked_x_attn[-1, :, :, :, :].unsqueeze(0)
            unmasked_attn_full = unmasked_attn_full[-1, :, :, :, :].unsqueeze(0)
            combined_mask = combined_mask[-1, :, :, :, :].unsqueeze(0)

        if self.use_last_head_distill:
            masked_x_attn = masked_x_attn[:, :, -1, :, :].unsqueeze(2)
            unmasked_attn_full = unmasked_attn_full[:, :, -1, :, :].unsqueeze(2)
            combined_mask = combined_mask[:, :, -1, :, :].unsqueeze(2)

        loss = self.decoder.kl_loss(masked_x_attn.log(), unmasked_attn_full)
        loss = loss.masked_fill(combined_mask, 0.0)
        if torch.any(loss.isnan()):
            import logging
            logging.info(f"There was nan loss in kl_loss not in mask_part.")

        dump_per_epoch = True
        if dump_per_epoch:
            from pathlib import Path
            import numpy as np
            import os
            from random import random
            # TODO switch to exp/.../ path
            if not str(self.epoch) in self.saved_examples:
                self.saved_examples[str(self.epoch)] = 0
            if self.saved_examples[str(self.epoch)] <= 4:
                if random() < 0.001:
                    path = Path(self.output_dir) / "debug_dumps" / str(self.epoch)
                    # B and decoder layers swap
                    unmasked_attn_full = torch.swapaxes(unmasked_attn_full, 0, 1)
                    masked_x_attn = torch.swapaxes(masked_x_attn, 0, 1)
                    combined_mask = torch.swapaxes(combined_mask, 0, 1)
                    loss = torch.swapaxes(loss, 0, 1)
                    for i, (uid, unmasked_attention, masked_attention) in enumerate(zip(utt_ids, unmasked_attn_full, masked_x_attn[:, :, :, :, :])):
                        savi = {
                            'random_block': self.current_random_blocks[i],
                            'masking_block': self.current_masking_blocks[i],
                            'utt_id': utt_ids[i],
                            'masking': combined_mask[i],
                            'kl_loss': loss[i],
                        }
                        os.makedirs(path / uid, exist_ok=True)
                        np.save(path / uid / "unmasked_attn.npy", unmasked_attention.detach().cpu().numpy())
                        np.save(path / uid / "masked_attn.npy", masked_attention.detach().cpu().numpy())
                        diff_attn = unmasked_attention - masked_attention
                        np.save(path / uid / "diff_attn.npy", diff_attn.detach().cpu().numpy())
                        np.save(path / uid / "masking_info.npy", savi)
                        self.saved_examples[str(self.epoch)] += 1

        # remove padding parts: encoder and decoder
        loss = loss.mean()

        return loss

    def _calc_tuple_loss(
            self,
            w_timing,
            use_libri_timings,
            utt_id,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        if use_libri_timings:
            print("Librispeech tuple loss not implemented yet")
            exit()
        else:
            w_timing = w_timing.float()
            for i, b in enumerate(w_timing):
                if not use_libri_timings:
                    start_times = [int(x.split('_')[1].split('-')[0]) for x in utt_id]
                    w_timing[i][w_timing[i] != -1] -= start_times[i]

                # Adjust timing labels for speed perturbation:
                if 'sp0.9-' in utt_id[i]:
                    w_timing[i][w_timing[i] != -1] /= 0.9
                elif 'sp1.1-' in utt_id[i]:
                    w_timing[i][w_timing[i] != -1] /= 1.1

        # 2. Compute attention loss
        target = w_timing[:, 1::2] # only end timings
        target_mask = target == -1
        loss_tuple = self.decoder.tuple_loss(self.decoder.tuple_loss_result.squeeze(), w_timing[:, 1::2])
        loss_tuple = loss_tuple.masked_fill(target_mask, 0.0).mean()
        return loss_tuple

    def _calc_timing_loss(self, masked_x_attn, w_timing, w_timing_lengths, text, text_lengths, utt_id, encoder_out_lens, use_libri_timings=False):
        # Padding mask
        padding_mask_encoder = make_pad_mask(encoder_out_lens)
        padding_mask_encoder = padding_mask_encoder.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(masked_x_attn.shape)

        dec_layers, batch_size, heads, dec_steps, enc_steps = masked_x_attn.shape
        # B x D x 1 (Label for correct encoder frame)
        att_labels = torch.zeros(1).unsqueeze(0).unsqueeze(0).repeat(batch_size, dec_steps, 1).to(w_timing.device)

        # H x B x D x E(1)
        att_labels_heads = torch.zeros(heads, *att_labels.shape)
        # B x H x D x E(1)
        att_labels_heads = att_labels_heads.swapaxes(0,1)

        w_timing = w_timing.float()
        for i, b in enumerate(w_timing):
            if not use_libri_timings:
                start_times = [int(x.split('_')[1].split('-')[0]) for x in utt_id]
                w_timing[i][w_timing[i] != -1] -= start_times[i]

            # Adjust timing labels for speed perturbation:
            if 'sp0.9-' in utt_id[i]:
                w_timing[i][w_timing[i] != -1] /= 0.9
            elif 'sp1.1-' in utt_id[i]:
                w_timing[i][w_timing[i] != -1] /= 1.1


            if not use_libri_timings:
                w_timing[i][w_timing[i] != -1] = w_timing[i][w_timing[i] != -1] // 4
                timings = w_timing[i][w_timing[i] != -1]
                # Loop over subword timings.
                # I think in swbd timings preprocessing already repeat for subwords
                end_timings = timings[1::2]
                att_labels_heads[i, -1, :len(end_timings), :] = end_timings.unsqueeze(-1)
                att_labels_heads[i, -1, len(end_timings):, :] = -1

            elif use_libri_timings:
                exit() # Not implemented yet
                last_timing = w_timing[i].squeeze() * 1000 / 40 # in s -> ms -> 40ms enc blocke


        # expand to decoder_layers && attention heads: DECODER_LAYERS x BATCH x HEADS x DECODER_STEPS x ENCODER STEPS
        # before: batch x heads x dec x enc
        att_labels_heads = att_labels_heads.unsqueeze(0).expand(dec_layers, *att_labels_heads.shape).to(masked_x_attn.device)
        att_labels_heads.requires_grad=False

        if self.only_last_layer_timing:
            masked_x_attn = masked_x_attn[-1, :, :, :, :].unsqueeze(0)
            att_labels_heads = att_labels_heads[-1, :, :, :, :].unsqueeze(0)
        if self.use_single_head:
            masked_x_attn = masked_x_attn[:, :, -1, :, :].unsqueeze(2)
            att_labels_heads = att_labels_heads[:,:,-1, :, :].unsqueeze(2)
        if self.only_last_timing:
            masked_x_attn = masked_x_attn[:,:,:,-1,:].unsqueeze(3)
            exit() # Not implemented
            # att_labels_heads = att_labels_heads[:,:,:,text_lengths[,:].unsqueeze(3)

        att_labels_heads = att_labels_heads.squeeze(-1) #L B H D
        att_labels_heads = torch.movedim(att_labels_heads, 1, 0) # B L H D
        masked_x_attn = torch.movedim(masked_x_attn, -1, 1) # L E B H D
        masked_x_attn = torch.swapaxes(masked_x_attn, 0, 2) # B E L H D
        loss = self.decoder.timing_loss(masked_x_attn.log(), att_labels_heads.long())
        loss = loss.mean()

        return loss

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        attn_cache_full_context: list = None,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Calc CTC loss
        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc
