import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.functional import softmax as sf


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()

        #here - initialize weights??

        self.W_h = nn.Linear(hidden_size,hidden_size, bias = False)       #need double inlet size since bidirectionsl?
        self.W_s = nn.Linear(hidden_size,hidden_size, bias = False)
        self.v = nn.Linear(hidden_size,1, bias = False)
        self.W_out = nn.Linear(2*hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size)
        encoder_outputs:(batch_size, max_src_len, hidden_size)
        src_lengths:    (batch_size)
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """

    

        #for each t produce an alignment vector 
        max_tgt_len = query.size()[1]
        max_src_len = encoder_outputs.size()[1]
        out_with_attention = torch.empty(query.size()[0], max_tgt_len,self.hidden_size )
        mask =  self.sequence_mask(src_lengths)



        #for each token 
        for t in range(max_tgt_len): 

            #include mask to not include padding tokens? 

            attention_scores = []
        
            st_min1 = query[:,t,:]
            WS = self.W_s(st_min1)

            #for each hidden state --> alignemnt score
            for i in range(max_src_len):


                WH = self.W_h(encoder_outputs[:,i,:])
                e_ti = self.v(torch.tanh(WS+WH))
                e_ti = e_ti.squeeze(-1)
                attention_scores.append(e_ti)

                #set attention scores to zero where padding occurs


        
            #normalize alignment scores - use softmax
            attention_scores_tensor = torch.stack(attention_scores, dim=1)
            #set attention values to zero where there is padding
            attention_scores_tensor = attention_scores_tensor.masked_fill(~mask, 0)

            normalized_attention_scores = sf(attention_scores_tensor, dim = 1)

            #compute context vector
            attention_scores_expanded = normalized_attention_scores.unsqueeze(-1)
            weighted_encoder_outputs = encoder_outputs * attention_scores_expanded
            ct = torch.sum(weighted_encoder_outputs, dim = 1)
            
            #concatenate context vector and s_tmin1
            concat = torch.cat((ct, st_min1), dim=1)  # Concatenate along the last dimension

            #compute st snake   
            s_snake_t = torch.tanh(self.W_out(concat))
        
            out_with_attention[:,t,:] = s_snake_t
            
        # print(f"src_lengths size = {src_lengths.size()}")

        # print(f"mask = {mask}")

       
        # #INCLUDE MASK!!!



        return out_with_attention

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        
        # apply embedding
        embedded = self.embedding(src)  
        # apply dropout -> does this apply dropout to each embedding vector?
        embedded = self.dropout(embedded)
        # pack the padded sequences
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass to LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack the packed sequence
        enc_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        enc_output = self.dropout(enc_output)

        final_hidden = (hidden, cell)

        return enc_output, final_hidden

        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)

class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # TODO: Implement the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(
        #         output,
        #         encoder_outputs,
        #         src_lengths,
        #     )
        #############################################
        
        # Initialize variables
        batch_size, max_tgt_len = tgt.size()
        outputs = []
        # Embed the input sequence
        embedded = self.embedding(tgt)  # (batch_size, max_tgt_len, hidden_size)
        embedded = self.dropout(embedded)

        if max_tgt_len == 1:
            # when only start token - how to handle this? 
            input_t = embedded[:, 0, :].unsqueeze(1)  # <SOS> token embedding
            output, dec_state = self.lstm(input_t, dec_state)  # First timestep
            outputs.append(output)
        else:
            # not only start tokem
            input_t = embedded[:, 0, :].unsqueeze(1)
            for t in range(1, max_tgt_len):
                output, dec_state = self.lstm(input_t, dec_state)
                outputs.append(output)
                input_t = embedded[:, t, :].unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_tgt_len, hidden_size)
        

        if self.attn is not None:
            outputs = self.attn(
                outputs,
                encoder_outputs,
                src_lengths,
            )


        outputs = self.dropout(outputs)

        return outputs, dec_state

        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)



class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
