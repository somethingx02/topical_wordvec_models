# -*- coding: utf8 -*-

# ====================Settings Used In yelp_preprocessor.py
# one side window size
HALF_WINDOW_SIZE = 5

# ====================Settings Used In training
ROOT_DIR = ('/media/Data1/'
            'topical_wordvec_models')

TRAIN_SET_PATH = '%s/datasets/train_sparseinstances.csv'\
    % ROOT_DIR

SAVE_DIR = '%s/save' % ROOT_DIR

# ----------Must be set before run
# word_embed size
HIDDEN_LAYER_SIZE = 100
# vocabulary size
VOCABULARY_SIZE = 8001

TRAINING_INSTANCES = 4631344

# topic count
# TOPIC_COUNT = 10
TOPIC_COUNT = 100

# dimension of encoder hidden layer (\pi)
DIM_ENCODER = 500

# CHUNK_SIZE = CHUNK_SIZE_MULTIPLIER_BATCH_SIZE * BATCH_SIZE


class DefaultConfig():
    '''
    default config for training parameters
    ====================
    params:
    ----------
    None

    return:
    ----------
    None
    '''
    # 256 best for 01 #if if 2048 then cuda memory exceeds
    batch_size = 128
    # 200 original
    epochs = 100
    # 0.0005 initial best,
    # learning rate initialize should depend on the batch size
    learning_rate = 0.0005
    # 0.93 0.005 best
    lr_decay = 0.9
    weight_decay = 1e-4
    model = 'TopicalWordEmbedding'

    # if this is false then run CPU
    # on_cuda = False
    # if this is false then run CPU
    on_cuda = True

    # tdnn_kernel=[(1,25),
    #             (2,50),
    #             (3,75),
    #             (4,100),
    #             (5,125),
    #             (6,150),
    #             (7,175)],
    # highway_size=700,
    # rnn_hidden_size=650,
    # dropout':0.0

    def set_attrs(self, kwargs):
        '''
        funciton for setting members
        ====================
        params:
        ----------
        kwargs: a dict

        return:
        ----------
        None
        '''
        for k, v in kwargs.items():
            # inbuilt function of python,
            # set the attributes of a class object.
            # For example: setattr(oDefaultConfig,epochs,50)
            # <=> oDefaultConfig.epochs = 50
            setattr(self, k, v)

    def get_attrs(self):
        '''
        the enhanced getattr,
        returns a dict whose key is public items in an object
        ======================
        params:
        ----------
        None

        return:
        ----------
        attrs: a dict whose key is public items in an object
        '''
        # attrs = dict()
        attrs = {}
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__')\
               and k != 'set_attrs' and k != 'get_attrs':
                # get the attr in an object
                attrs[k] = getattr(self, k)
        return attrs


if __name__ == '__main__':

    config = DefaultConfig()
    print(config.get_attrs())
    config.set_attrs({'epochs': 200, 'batch_size': 64})
    print(config.get_attrs())
