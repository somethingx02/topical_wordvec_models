# -*- coding: utf8 -*-

# torch is a file named torch.py
import torch
# torch here is a folder named torch
from torch.autograd import Variable
from torchnet import meter
# this is filename, once imported, you can use the classes in it

from models import topicalWordEmbedding

# equal to from models.topicalAttentionGRU import TopicalAttentionGRU
from settings import HALF_WINDOW_SIZE
from settings import HIDDEN_LAYER_SIZE
from settings import VOCABULARY_SIZE
from settings import TRAINING_INSTANCES
from settings import TOPIC_COUNT
from settings import DIM_ENCODER

from settings import DefaultConfig

from utils import *

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# cuda device id
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train(**kwargs):
    '''
    begin training the model
    *kwargs: train(1, 2, 3, 4, 5) =>
    kwargs[0] = 1 kwargs[1] = 2 ..., kwargs is principally a tuple
    **kwargs: train(a=1, b=2, c=3, d=4)
    CustomPreProcessor =>
    kwargs[a] = 1, kwargs[b] = 2, kwargs[c] = 3,
    kwargs[d] = 4, kwargs is principally a dict
    function containing kwargs *kwargs **kwargs must be written as:
    def train(args,*args,**args)
    '''

    saveid = latest_save_num() + 1
    # the save_path is
    save_path = '%s/%d' % (SAVE_DIR, saveid)
    print("logger save path: %s" % (save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path_each_save = '%s/log.txt' % save_path
    model_path_each_save = '%s/model' % save_path
    logger = get_logger(log_path_each_save)

    config = DefaultConfig()

    # settings here, avalid_data_utillso about whether on cuda
    config.set_attrs(kwargs)
    # print(config.get_attrs())

    epochs = config.epochs
    batch_size = config.batch_size

    # determine whether to run on cuda
    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if not config.on_cuda:
            logger.info('Cuda is unavailable,\
                Although wants to run on cuda,\
                Model still run on CPU')

    # 300 in our model
    # 1024 is the Elmo size,
    # the concatenated hidden size is supposed to Elmo size,
    # however, any size is OK
    # it depends on the setting
    # attention size should be a smoothed representation of character-emb

    if config.model == 'TopicalWordEmbedding':
        model = topicalWordEmbedding.TopicalWordEmbedding(
            param_on_cuda=config.on_cuda,
            param_half_window_size=HALF_WINDOW_SIZE,
            param_vocabulary_size=VOCABULARY_SIZE,
            param_hidden_layer_size=HIDDEN_LAYER_SIZE,
            param_encoder_pi_size=DIM_ENCODER,
            param_topic_count=TOPIC_COUNT)

    if config.on_cuda:
        logger.info('Model run on GPU')
        model = model.cuda()
        logger.info('Model initialized on GPU')
    else:
        logger.info('Model run on CPU')
        model = model.cpu()
        logger.info('Model initialized on CPU')

    # print('logger-setted',file=sys.stderr)
    # output the string informetion to the log
    logger.info(model.modelname)
    # output the string information to the log
    logger.info(str(config.get_attrs()))

    # read in the trainset and the trial set
    # Train Set
    train_data_manager = DataManager(batch_size, TRAINING_INSTANCES)

    train_data_manager.load_dataframe_from_file(TRAIN_SET_PATH)

    # set the optimizer parameter,
    # such as learning rate and weight_decay,
    # function Adam, a method for Stochastic Optizimism

    # load the learning rate in config, that is settings.py
    lr = config.learning_rate
    # params_iterator_requires_grad can only be iterated once
    params_iterator_requires_grad = filter(
        lambda trainingParams: trainingParams.requires_grad,
        model.parameters())
    # print(len(list(params_iterator_requires_grad)))

    # 25 parameters
    # weight decay that is L2 penalty that is L2 regularization,
    # usually added after a cost function(loss function),
    # for example C=C_0+penalty, QuanZhongShuaiJian,
    # to avoid overfitting
    optimizer = torch.optim.Adam(
        params_iterator_requires_grad,
        lr=lr,
        weight_decay=config.weight_decay)

    # By default, the losses are averaged over observations
    # for each minibatch.
    # However, if the field size_average is set to False,
    # the losses are instead
    # summed for each minibatch

    # The CrossEntropyLoss,
    # My selector in my notebook = loss + selecting strategy
    # (often is selecting the least loss)

    # criterion = torch.nn.CrossEntropyLoss(size_average=False)

    # once you have the loss function, you also have
    # to train the parameters in g(x),
    # which will be used for prediction

    # the loss calculated after the smooth method,
    # that is L2 penalty mentioned in torch.optim.Adam
    loss_meter = meter.AverageValueMeter()
    # get confusionMatrix, the confusion matrix is the one show as follows:
    # confusion_matrix = meter.ConfusionMeter(
    #     CLASS_COUNT)
    '''                    class1 predicted class2 predicted class3 predicted
    class1 ground truth  [[4,               1,               1]
    class2 ground truth   [2,               3,               1]
    class2 ground truth   [1,               2,               9]]
    '''
    model.train()
    # pre_loss = 1e100
    # best_acc = 0
    smallest_loss = 0x7fffffffffffffffffffffffffffffff

    for epoch in range(epochs):
        '''
        an epoch, that is, train data of all
        barches(all the data) for one time
        '''

        loss_meter.reset()
        # confusion_matrix.reset()

        train_data_manager.reshuffle_dataframe()

        # it was ceiled, so it is "instances/batch_size + 1"
        n_batch = train_data_manager.n_batches()

        batch_index = 0
        for batch_index in range(0, n_batch - 1):
            # this operation is time consuming
            xn, wc = train_data_manager.next_batch()

            # long seems to trigger cuda error,
            # it cannot handle long
            # variable by defalut requires_grad = False
            # t = torch.Tensor(1)
            # t.to(torch.float32) <=> t.float()
            # t.to(torch.int64) <=> t.long()
            var_xn = Variable(torch.from_numpy(xn).float())
            # print( x.size() )
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False)
            # y = y - 1
            # print(y.size())

            # #########################logger.info('Begin fetching a batch')
            loss = eval_batch(model, var_xn, var_wc, config.on_cuda)
            # #########################logger.info(
            #     'End fetching a batch, begin optimizer')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # #########################logger.info('End optimizer')
            # data is the tensor,
            # [0] is a Python number,
            # if a 0-dim tensor, then .item will get the python number,
            # if 1-dim then .items will get list
            loss_meter.add(loss.data.item())

            # confusion_matrix.add(scores.data, y.data)
            # if batch_index == 10 then display the accuracy of the batch
            if (batch_index + 1) % 200 == 0:
                # for 2 LongTensors,  17 / 18 = 0
                # accuracy = corrects.float() / config.batch_size
                # .value()[0] is the loss value
                logger.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f' % (
                    epoch, epochs, batch_index, n_batch,
                    loss_meter.value()[0]))
        # abandon the tail batch, because it will trigger duplicate
        # context window thus causing loss == nan
        if TRAINING_INSTANCES % batch_size == 0:
            train_data_manager.set_current_cursor_in_dataframe_zero()
        else:
            # train_data_manager.set_current_cursor_in_dataframe_zero()
            print('!!!!!!!!!!!Enter tail batch')
            # the value can be inherented
            batch_index += 1
            (xn, wc) = train_data_manager.tail_batch_nobatchpadding()
            # long seems to trigger
            # t = torch.Tensor(1)
            # t.to(torch.float32) <=> t.float()
            # t.to(torch.int64) <=> t.long()
            var_xn = Variable(torch.from_numpy(xn).float())
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False)
            # y = y - 1
            loss = eval_batch(model, var_xn, var_wc, config.on_cuda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.data.item())
            # confusion_matrix.add( scores.data , y.data )
            # if batch_index == 10 then display the accuracy of the batch
            if (batch_index + 1) % 200 == 0:
                # for 2 LongTensors,  17 / 18 = 0
                # accuracy = corrects.float() / config.batch_size
                # print("accuracy = %f, corrects = %d"%(accuracy, corrects))
                # .value()[0] is the loss value y = Variable(
                #     torch.LongTensor(y), requires_grad = False)
                logger.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f' % (
                    epoch, epochs, batch_index, n_batch,
                    loss_meter.value()[0]))
        # print('!!!!!!!!!!!Exit tail batch')
        # after an epoch it should be evaluated
        # switch to evaluate model
        model.eval()
        # if (batch_epochsindex + 1) % 25 == 0:
        # every 50 batches peek its accuracy and get the best accuracy
        # confusion_matrix_value=confusion_matrix.value()
        # acc = 0
        # for i in range(CLASS_COUNT):
        #     correct prediction count
        #     acc += confusion_matrix_value[i][i]
        # the accuracy, overall accuracy in an epoch
        # acc = acc / confusion_matrix_value.sum()

        # a 1-dim tensor with lenth 1,
        # so you have to access the element by [0]
        # loss_meter.value() = (mean, var), mean is the average among batches
        the_overall_averaged_loss_in_epoch = loss_meter.value()[0]
        logger.info('epoch: %d/%d\taverage_loss: %f' % (
            epoch, epochs, the_overall_averaged_loss_in_epoch))
        # switch to train model
        model.train()

        # if accuracy increased, then save the model and
        # change the learning rate
        if loss_meter.value()[0] < smallest_loss:
            # save the model
            model.save(model_path_each_save)
            logger.info('model saved to %s' % model_path_each_save)

            # change the learning rate
            if epoch < 4:
                lr = lr * config.lr_decay
            else:
                if epoch < 8:
                    lr = lr * 0.97
                else:
                    lr = lr * 0.99
            logger.info('learning_rate changed to %f' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            smallest_loss = loss_meter.value()[0]
        else:
            print('the loss_meter = ', loss_meter.value()[0])

        # pre_loss = loss_meter.value()[0]


def eval_batch(model, var_xn, var_wc, on_cuda):
    '''
    evaluate the logits of each instance, loss, corrects in a batch
    '''
    if on_cuda:
        var_xn, var_wc = var_xn.cuda(), var_wc.cuda()
    else:
        var_xn, var_wc = var_xn.cpu(), var_wc.cpu()

    # batch_size * dim
    var_xnwc = torch.cat((var_xn, var_wc),
                         dim=1)
    nll_term, kld_term = model(var_xnwc)

    # since the size_average parameter == False,
    # the loss is the sumed loss of the batch.
    # The loss is a value rather than a vector

    # CrossEntropyLoss takes in a
    # vector and a class num (usually a index num of the vector)
    # loss = criterion(logits, y)

    loss = torch.sum(nll_term + kld_term, dim=0)

    # [0] : max value of dim 1 [1]: max index of dim 1 LongTensor
    # model_training_predicts = torch.max( logits , 1)[ 1 ]

    # assert model_training_predicts.size() == y.size()

    # y.data shouldn't contain -1 or 5,
    # or will trigger cuda error in loss = criterion(logits, y)
    # but will display later
    # corrects is a LongTensor sotred in cuda,
    # y.data means the tensor of the variable
    # corrects = (model_training_predicts.data == y.data).sum()
    # print('-----------------%d' % (corrects))

    return loss


# def compute_elmo_rep(model_dir, input_list,
#                      mtype='TopicalWordEmbedding'):
#     '''
#     Given a list of documents,
#     return a list of embedded documents
#     each element in list is [sentence len] * [word embedding dim]
#     '''
#     # Just take the default config to do the prediction work
#     config = DefaultConfig()
#     config.set_attrs({'batch_size': 8})
#     model_path = '%s/model' % model_dir

#     text_processor = TextPreProcessor(
#     normailze = ['url','email','percent','money','phone','user',
#         'time','date','number'],
#     annotate = {"hashtag","allcaps","elongated","repeated",
#         "emphasis","censored"},
#     fix_html = True,
#     segmenter = "english",
#     corrector = "english",
#     unpack_hashtags = True,
#     unpack_contractions = True,
#     spell_correct_elong = False,

#     tokenizer = SocialTokenizer(lowercase = True).tokenize,
#     dicts = [emoticons])

#     listTokenized = list(text_processor.pre_process_docs( input_list ) )
#     print('After tokenization:')
#     print(listTokenized)

#     tensorTokenizedCharEncoded = batch_to_ids( listTokenized )#[ ['I', 'am', 'a' ,'sentense'] , ['A','sentense'] ] )#listShuffledReviewsTokenized )
#     # print( listShuffledReviewsCharacterEmbedded[0].size() )

#     arrayTokenizedCharEncoded = tensorTokenizedCharEncoded.numpy().astype(numpy.int32)

#     x = Variable( torch.from_numpy(arrayTokenizedCharEncoded).long(), requires_grad=False)

#     if config.on_cuda:
#         x = x.cuda()
#     else:
#         x = x.cpu()

#     #print(x.size())


#     model=topicalWordEmbedding.TopicalWordEmbedding(
#            # param_on_cuda=config.on_cuda,
#            # param_half_window_size=HALF_WINDOW_SIZE,
#            # param_vocabulary_size=VOCABULARY_SIZE,
#            # param_hidden_layer_size=HIDDEN_LAYER_SIZE,
#            # param_encoder_pi_size=DIM_ENCODER,
#            # param_topic_count=TOPIC_COUNT)
#     print('Loading trained model')

#     # here, load and save are defined in biLSTMAttention.py
#     # load <=> model.load_state_dict( torch.load(path) )
#     # save <=> torch.save( model.state_dict(), path )

#     # an other way:
#     # model = torch.load( path )
#     # has 2 field, if torch.save( model, path ),
#     # then both ['state_dict'] and ['struct'] != None
#     # torch.save( model, path )

#     if config.on_cuda:
#         model.load( model_path )
#         model=model.cuda()
#     else:
#         model.load_cpu_from_gputrained( model_path )
#         model=model.cpu()

#     elmo_dict = model.forward_obtainTrainedElmoRep(x)

#     # since num_output_representations = 1, so len(list_elmo_rep) = 1,
#     elmo_rep = elmo_dict['elmo_representations'][0]
#     # if num_output_representations == 2,
#     # then will produce 2 same elmo_representations of
#     # [batch_size, seq_len, wordembedding_len]

#     #print(elmo_rep.size())
#     arr_elmo_rep = elmo_rep.data.cpu().numpy()

#     return arr_elmo_rep


if __name__ == '__main__':

    # ==========Train with the training set
    train()

    # #==========Predict with the testing set
    # predict('%s/6'%SAVE_DIR)

    # #==========Calculate with the predicted result
    # calculate_accuracy('%s/0'%SAVE_DIR)

