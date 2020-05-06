# -*- coding: utf8 -*-

import torch

# torch.nn and torch two different module from torch
import torch.nn
import torch.nn.functional
from torch.distributions.normal import Normal

import torch.autograd

import os

# from vaeEncoder import VaeEncoder
# from vaeDecoder import VaeDecoder
from models.vaeEncoder import VaeEncoder
from models.vaeDecoder import VaeDecoder

import numpy as np


class TopicalWordEmbedding(torch.nn.Module):
    '''
    You have to inherent nn.Module to define your own model
    two virtual functions, loda and save must be instantialized
    '''

    def __init__(
            self,
            param_on_cuda,
            param_half_window_size,
            param_vocabulary_size,
            param_hidden_layer_size,
            param_encoder_pi_size,
            param_topic_count):
        '''
        initialise the TopicalWordEmbedding
        ====================
        params:
        ----------
        param_half_window_size: C
        param_vocabulary_size: xn, wc size
        param_hidden_layer_size: z size
        param_encoder_pi_size: pi size of encoder
        param_topic_count: topic count

        return:
        ----------
        None
        '''

        super(TopicalWordEmbedding, self).__init__()
        # same with the class name
        self.modelname = 'TopicalWordEmbedding'
        self.on_cuda = param_on_cuda

        self.half_window_size = param_half_window_size
        self.vocabulary_size = param_vocabulary_size
        self.hidden_layer_size = param_hidden_layer_size
        self.encoder_pi_size = param_encoder_pi_size
        self.topic_count = param_topic_count

        self.vae_decoder = VaeDecoder(
            param_dim_topic=param_topic_count,
            param_dim_vocab=param_vocabulary_size,
            param_dim_hidden=param_hidden_layer_size)
        self.vae_encoder = VaeEncoder(
            param_dim_encoder=param_encoder_pi_size,
            param_dim_vocab=param_vocabulary_size,
            param_dim_hidden=param_hidden_layer_size)
        self.standard_normal = Normal(
            loc=torch.zeros(param_hidden_layer_size),
            scale=1.0)

        return None

    def load(self, path):
        '''
        cpu => cpu or
        gpu => gpu
        lode state dict
        '''
        self.load_state_dict(torch.load(path))

    def save(self, path):
        '''
        save state dict
        '''
        save_result = torch.save(self.state_dict(), path)
        return save_result

    def load_cpu_from_gputrained(self, path):
        '''
        load all trained model to cpu
        '''
        self.load_state_dict(
            torch.load(path, map_location='cpu'))

    def sample_an_zs(self, param_mu, param_sigma_log_pow):
        '''
        generate one sample of z
        =====================
        params:
        ----------
        param_mu: mu
        param_sigma_log_pow: sigma_log

        return:
        ----------
        z_s: one sample of z

        using: diagnol x vec <=> diagnolvec .*
        '''
        # create an empty tensor with specific size
        # eps = norma
        (batch_size, hidden_layer_size) = param_mu.size()
        if self.on_cuda:
            # eps = torch.autograd.Variable(
                # self.standard_normal.sample()).cuda()
            eps = torch.autograd.Variable(
                self.standard_normal.sample(
                    sample_shape=(batch_size,))).cuda()
        else:
            eps = torch.autograd.Variable(
                self.standard_normal.sample(
                    sample_shape=(batch_size,)))
        sigma = torch.sqrt(torch.exp(param_sigma_log_pow))
        z_s = param_mu + sigma * eps
        return z_s

    def forward(self, param_input_xnwc):
        '''
        from input to output
        ====================
        params:
        ----------
        param_input_xnwc: a batchsize, 2 * VOCASIZE tensor

        return:
        ----------
        nll_term, kld_term
        '''
        (batch_size, vocabulary_size_x2) = param_input_xnwc.size()
        assert self.vocabulary_size * 2 == vocabulary_size_x2

        mu, sigma_log_pow = self.vae_encoder(param_input_xnwc)
        z_s = self.sample_an_zs(mu, sigma_log_pow)
        p_xn, p_wc = self.vae_decoder(z_s)
        # each batch size apply
        kld_term = -0.5 * torch.sum(
            1 + sigma_log_pow - mu**2 - torch.exp(sigma_log_pow),
            dim=1)
        xn = param_input_xnwc[:, 0:self.vocabulary_size]
        wc = param_input_xnwc[:, self.vocabulary_size:]
        xn_sum_log_pow = torch.sum(
            torch.log(torch.pow(p_xn, xn)),
            dim=1)
        wc_sum_log_pow = torch.sum(
            torch.log(torch.pow(p_wc, wc)),
            dim=1)
        xnwc_sum_log_pow = xn_sum_log_pow + wc_sum_log_pow
        nll_term = -xnwc_sum_log_pow

        return nll_term, kld_term

    def forward_obtain_xn_rep(
            self,
            param_input_xn, param_input_wc):
        '''
        comput the pivot xn representation given xn and wc
        ====================
        params:
        ----------
        param_input_xn: pivot xn
        param_input_wc: window wc

        return:
        ----------
        an embedding of size |z|
        it is the maximum likelihood of |z|,
        that is, \mu
        '''
        xnwc = torch.cat((param_input_xn, param_input_wc),
                         dim=1)
        mu, sigma_log_pow = self.vae_encoder(xnwc)
        return mu

    def forward_obtain_xn_klrep(
            self,
            param_input_xn, param_input_wc):
        '''
        comput the pivot xn representation given xn and wc
        ====================
        params:
        ----------
        param_input_xn: pivot xn
        param_input_wc: window wc

        return:
        ----------
        an embedding of size |z|
        it is the maximum likelihood of |z|,
        that is, \mu,\sigma_log_pow
        '''
        xnwc = torch.cat((param_input_xn, param_input_wc),
                         dim=1)
        mu, sigma_log_pow = self.vae_encoder(xnwc)
        return mu, sigma_log_pow

    def forward_obtain_xn_zeta(
            self,
            param_input_xn, param_input_wc):
        '''
        comput the pivot xn's zeta representation given xn and wc
        ====================
        params:
        ----------
        param_input_xn: pivot xn
        param_input_wc: window wc

        return:
        ----------
        an embedding of size |\zeta|
        it is the maximum likelihood of |z|,
        and transfered to \zeta
        that is, \mu
        '''
        xnwc = torch.cat((param_input_xn, param_input_wc),
                         dim=1)
        mu, sigma_log_pow = self.vae_encoder(xnwc)
        zeta = self.vae_decoder.forward_obtain_zeta(mu)
        return zeta


if __name__ == '__main__':

    # cuda device id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    # torch.LongTensor(128, 23, 50)
    # var_input = torch.autograd.Variable(
    #     torch.ones([16, 1219, 50], dtype=torch.long))
    # var_input = var_input * 261

    ar_input = np.zeros((3, 4 * 2), dtype=np.int32)
    ar_input[0, 1] = 1
    ar_input[1, 3] = 1
    ar_input[2, 2] = 1
    ar_input[0, 4 + 1] = 2
    ar_input[1, 4 + 3] = 3
    ar_input[2, 4 + 2] = 4
    ar_input[0, 4 + 0] = 4
    ar_input[1, 4 + 2] = 3
    ar_input[2, 4 + 1] = 2
    var_input = torch.autograd.Variable(
        torch.Tensor(ar_input))

    att_model_test = TopicalWordEmbedding(
        param_on_cuda=True,
        param_half_window_size=5,
        param_vocabulary_size=4,
        param_hidden_layer_size=6,
        param_encoder_pi_size=7,
        param_topic_count=8)

    # res = att_model_test(var_input)
    att_model_test = att_model_test.cuda()
    var_input = var_input.cuda()

    var_output_nll, var_output_kld = att_model_test(var_input)
    print(var_output_nll.size())
    print(var_output_kld.size())

    mu = att_model_test.forward_obtain_xn_rep(
        torch.autograd.Variable(
            torch.FloatTensor([[0, 0, 1, 0],
                               [1, 0, 0, 0]])).cuda(),
        torch.autograd.Variable(
            torch.FloatTensor([[0, 3, 1, 0],
                               [1, 2, 6, 2]])).cuda())
    print(mu.size())

    zeta = att_model_test.forward_obtain_xn_zeta(
        torch.autograd.Variable(
            torch.FloatTensor([[0, 0, 1, 0],
                               [1, 0, 0, 0]])).cuda(),
        torch.autograd.Variable(
            torch.FloatTensor([[0, 3, 1, 0],
                               [1, 2, 6, 2]])).cuda())
    print(zeta.size())
    print(att_model_test.vae_decoder.MATRIX_decoder_beta.size())

    # #====================Testing center
    # #----------Test for Normal
    # normal_dist = Normal(loc=torch.zeros([3]),
    #                      scale=1.0)
    #                      #scale=torch.ones([2]))
    # print(normal_dist.sample())
