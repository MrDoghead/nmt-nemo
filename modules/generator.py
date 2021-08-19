import torch
import numpy as np
from modules.trt_helper import DecWrapper
from modules.embedding import TransformerEmbedding

NEG_INF = -10000.0
device = torch.device('cpu')

def mask_padded_tokens(tokens, pad_id):
    mask = np.array((tokens != pad_id),dtype=np.int32)
    #mask = tokens != pad_id
    return mask

class BeamSearchGenerator:
    def __init__(
            self,
            vocab_size,
            hidden_size,
            max_sequence_length,
            dec_trt_path,
            pad=0,
            bos=1,
            eos=2,
            max_sequence_length=512,
            max_delta_length=20,
            batch_size=1,
            beam_size=4,
            len_pen=0.6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.dec_trt_path = dec_trt_path
        self.pad, self.bos, self.eos = pad, bos, eos
        self.max_seq_length = max_sequence_length
        self.max_delta_len = max_delta_length
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.len_pen = len_pen
        self.embedding = TransformerEmbedding(
            vocab_size = self.vocab_size,
            hidden_size = self.hidden_size,
            max_sequence_length = self.max_sequence_length,
            num_token_types = 2,
            learn_positional_encodings = False,
        )
        self.embedding.restore_from(chk_path='./model_bin/model_weights.ckpt')

    @staticmethod
    def compute_len_penalty(lengths, alpha):
        """Returns length penalty according to https://arxiv.org/pdf/1609.08144.pdf"""
        return ((5 + lengths) / 6).pow(alpha)

    def _prepare_for_search(self, decoder_input_ids, encoder_hidden_states):
        batch_size = self.batch_size

        # for encoder-decoder generation, maximum length of generated sequence
        # is min(max_sequence_length, src_len + max_delta_length)
        if encoder_hidden_states is not None:
            batch_size, src_len, _ = encoder_hidden_states.shape
            max_seq_length = min(self.max_seq_length, src_len + self.max_delta_len)
        else:
            max_seq_length = self.max_seq_length

        # if no input is provided, start with the batch of <bos> tokens
        if decoder_input_ids is not None:
            tgt = decoder_input_ids
            batch_size, tgt_len = decoder_input_ids.size()
        else:
            tgt = torch.zeros(batch_size, 1).long().fill_(self.bos).to(device)
            tgt_len = 1
        max_generation_length = max_seq_length - tgt_len

        return tgt, batch_size, max_generation_length

    def log_softmax(self, hidden_states):
        chk_path = './model_bin/model_weights.ckpt'
        chk = torch.load(chk_path)
        weight = chk['log_softmax.mlp.layer0.weight'].to(device)
        bias = chk['log_softmax.mlp.layer0.bias'].to(device)

        mlp_layer0_out = torch.matmul(hidden_states, weight.transpose(0,1)) \
                + bias.view(1,1,-1)
        log_probs = torch.log_softmax(mlp_layer0_out, dim=-1)
        return log_probs

    def _one_step_forward(
            self,
            decoder_input_ids=None,
            encoder_hidden_states=None,
            encoder_input_mask=None,
            decoder_mems_list=None,
            pos=0
        ):
        """
        One step of autoregressive output generation.
        Args:
            decoder_input_ids: starting sequence of tokens to generate from;
                if None, generation will start from a batch of <bos> tokens
            encoder_hidden_states: output of the encoder for conditional
                sequence generation; if None, generator will use unconditional
                mode (e.g., language modeling)
            encoder_input_mask: input mask used in the encoder
            decoder_mems_list: list of size num_layers with cached activations
                of sequence (x[1], ..., x[k-1]) for fast generation of x[k]
            pos: starting position in positional encoding
        """
        decoder_hidden_states = self.embedding(decoder_input_ids, start_pos=pos)
        # convert torch to numpy to match the format of trt input
        decoder_input_ids = decoder_input_ids.numpy().astype(np.int32)
        encoder_hidden_states = encoder_hidden_states.numpy()
        encoder_input_mask = encoder_input_mask.numpy().astype(np.int32)

        decoder_input_mask = mask_padded_tokens(decoder_input_ids, self.pad)
        batch_size, _, hidden_size = encoder_hidden_states.shape
        shape_of_output = (batch_size, pos+1, hidden_size)

        # tensorRT decoder engine
        print('===== decoder =====')
        print('decoder_input_ids:',decoder_input_ids)
        print('decoder_input_mask:',decoder_input_mask)
        print('encoder_hidden_states:',encoder_hidden_states)
        print('encoder_hidden_states.shape:',encoder_hidden_states.shape)
        print('encoder_input_mask:',encoder_input_mask)
        print('encoder_input_mask.shape',encoder_input_mask.shape)
        print('shape_of_output:',shape_of_output)
        print('batch size:',batch_size)
        dec_wrapper = DecWrapper(trt_path=self.dec_trt_path)
        dec_output = dec_wrapper.do_inference(
            decoder_input_ids,
            decoder_input_mask,
            encoder_hidden_states,
            encoder_input_mask,
            shape_of_output,
            batch_size
        )
        decoder_mems_list = dec_output[0].reshape(shape_of_output)
        print('decoder output:',decoder_hidden_states)
        # convert back to tensor
        decoder_mems_list = torch.from_numpy(decoder_mems_list).to(device)
        log_probs = self.log_softmax(hidden_states=decoder_mems_list)
        return log_probs, decoder_mems_list

    def forward(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None):
        tgt, batch_size, max_generation_length = self._prepare_for_search(decoder_input_ids, encoder_hidden_states)

        # generate initial buffer of beam_size prefixes-hypotheses
        log_probs, decoder_mems_list = self._one_step_forward(
                decoder_input_ids = tgt, 
                encoder_hidden_states = encoder_hidden_states, 
                encoder_input_mask = encoder_input_mask, 
                decoder_mems_list = None,
                pos = 0)
        # [bs,1,32000] -> [bs,32000,1] -> [bs,4,1]
        # [bs,4,1] -> [4*bs,1]
        scores, prefixes = torch.topk(log_probs.permute(0, 2, 1), self.beam_size, dim=1)
        scores, prefixes = scores.view(-1, 1), prefixes.view(-1, 1)
        # repeat init target prefixes and cached memory states beam_size times
        # [bs,1] -> [bs,4] -> [4*bs,1] cat [4*bs,1] -> [4*bs,2]
        # [bs,1,1024] -> [4*bs,1,1024]
        prefixes = torch.cat((tgt.repeat(1, self.beam_size).view(-1, 1), prefixes), dim=1)
        for j in range(len(decoder_mems_list)):
            decoder_mems_list[j] = decoder_mems_list[j].repeat(self.beam_size, 1, 1)

        # repeat source sequence beam_size times for beam search
        # [bs,seq] -> [4*bs,seq]
        # [bs,seq,1024] -> [4*bs,seq,1024]
        if encoder_hidden_states is not None:
            _, src_length, hidden_size = encoder_hidden_states.size()
            encoder_input_mask = encoder_input_mask.repeat(1, self.beam_size).view(-1, src_length)
            encoder_hidden_states = encoder_hidden_states.repeat(1, self.beam_size, 1).view(
                -1, src_length, hidden_size
            )
        else:
            hidden_size = decoder_mems_list[0].size(2)

        # pad_profile tracks finished hypotheses to generate only <pad> tokens
        # if <eos> or <pad> has been generated
        pad_profile = torch.zeros_like(scores).long().to(device)

        # prefixes_len tracks lengths of generated hypotheses to perform
        # length penalty correction
        prefixes_len = torch.zeros_like(scores).fill_(prefixes.size(1) + 1)

        for i in range(max_generation_length):

            # mask all finished hypotheses to exclude them from beam
            pad_mask = pad_profile.repeat(1, self.beam_size)

            # generate and score candidates for prefixes continuation
            log_probs, decoder_mems_list = self._one_step_forward(
                decoder_input_ids = prefixes[:, -1:], 
                encoder_hidden_states = encoder_hidden_states, 
                encoder_input_mask = encoder_input_mask, 
                decoder_mems_list = decoder_mems_list,
                pos = i + 1
            )
            scores_i, prefixes_i = torch.topk(log_probs[:, -1, :], self.beam_size, dim=-1)

            # for all prefixes ending with <eos> or <pad> replace generated
            # continuations with <pad>
            prefixes_i = self.pad * pad_mask + prefixes_i * (1 - pad_mask)

            # force all hypotheses but one generated from already finished
            # hypotheses to have extremely low score, so they will not be
            # considered during beam re-ranking
            pad_mask[:, 1:] = pad_mask[:, 1:] * NEG_INF
            scores = scores + scores_i * (1 - pad_mask).to(scores.dtype)

            # choose top-k hypotheses with length penalty applied
            len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
            scores = scores / len_penalties
            scores, indices_i = torch.topk(scores.view(-1, self.beam_size ** 2), self.beam_size, dim=1)
            scores = scores.view(-1, 1) * len_penalties

            # select prefixes which correspond to the chosen hypotheses
            prefixes = prefixes.unsqueeze(1).repeat(1, self.beam_size, 1)
            prefixes = torch.cat((prefixes, prefixes_i.unsqueeze(2)), dim=2)
            prefixes = prefixes.view(batch_size, self.beam_size ** 2, -1)
            p_len = prefixes.size(2)
            prefixes_ids = indices_i.unsqueeze(2).repeat(1, 1, p_len)
            prefixes = prefixes.gather(1, prefixes_ids).view(-1, p_len)

            # reshuffle cached decoder memory states to restore the order
            # of hypotheses broken after top-k selection
            mems_ids = indices_i.unsqueeze(2).unsqueeze(3).repeat(1, 1, p_len - 1, hidden_size) // self.beam_size
            for j in range(len(decoder_mems_list)):
                decoder_mems_list[j] = (
                    decoder_mems_list[j]
                    .view(-1, self.beam_size, p_len - 1, hidden_size)
                    .gather(1, mems_ids)
                    .view(-1, p_len - 1, hidden_size)
                )

            # update prefixes_len and pad_profile
            not_eos_pad = prefixes.ne(self.eos) & prefixes.ne(self.pad)
            prefixes_len = 1 + not_eos_pad.sum(dim=1, keepdim=True).to(scores.dtype)
            pad_profile = (~not_eos_pad[:, -1:]).long()

            # if all hypotheses end with <eos> or <pad>, interrupt search
            if pad_profile.sum() == batch_size * self.beam_size:
                break

    # select best performing hypotheses in each element of the batch
    len_penalties = self.compute_len_penalty(prefixes_len, self.len_pen)
    scores = scores / len_penalties
    best_guesses = (
        torch.argmax(scores.view(-1, self.beam_size), dim=1, keepdim=True).repeat(1, prefixes.size(1)).unsqueeze(1)
    )
    tgt = prefixes.view(batch_size, self.beam_size, -1).gather(1, best_guesses)

    return tgt.squeeze(1)