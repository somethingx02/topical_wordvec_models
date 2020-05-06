# -*- coding: utf8 -*-

def predict(model_dir, mtype='TopicalWordEmbedding'):
    '''
    load the model and conduct the prediction,
    the prediction is added with 1 since the
    original prediction is the index
    prediction is saved in '%s/0'%SAVE_DIR
    '''

    model_path = '%s/model' % model_dir
    output_path = '%s/res.txt' % model_dir

    # Just take the default config to do the prediction work
    config = DefaultConfig()
    config.set_attrs({'batch_size': 128})

    if mtype == 'topicalWordEmbedding':
        model = topicalWordEmbedding.TopicalWordEmbedding(
            param_on_cuda=config.on_cuda,
            param_half_window_size=HALF_WINDOW_SIZE,
            param_vocabulary_size=VOCABULARY_SIZE,
            param_hidden_layer_size=HIDDEN_LAYER_SIZE,
            param_encoder_pi_size=DIM_ENCODER,
            param_topic_count=TOPIC_COUNT)
    print('Loading trained model')

    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if not config.on_cuda:
            print('Cuda is unavailable, \
                Although wants to run on cuda, \
                Model still run on CPU')

    # here, load and save are defined in biLSTMAttention.py
    # load <=> model.load_state_dict( torch.load(path) )
    # save <=> torch.save( model.state_dict(), path )

    # an other way:
    # model = torch.load( path )
    # has 2 field, if torch.save( model, path ),
    # then both ['state_dict'] and ['struct'] != None
    # torch.save( model, path )

    if config.on_cuda:
        model.load(model_path)
        model = model.cuda()
    else:
        model.load_cpu_from_gputrained(model_path)
        model = model.cpu()

    print('Begin loading data')
    # the batch_size makes no differences
    datamanager = DataManager(
        param_batch_size=config.batch_size,
        param_training_instances_size=TESTING_INSTANCES)
    datamanager.load_dataframe_from_file( TEST_SET_PATH )
    n_batch = datamanager.n_batches()
    res= numpy.array([])

    batch_index = 0

    for batch_index in range(n_batch - 1):
        ( x , y ) = datamanager.next_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        scores = model.forward(x)
        _ , predict = torch.max(scores, 1) # predict is the first dimension , its the same as [ 1 ] 

        res = numpy.append( res, predict.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )

    if TESTING_INSTANCES % config.batch_size == 0:
        datamanager.set_current_cursor_in_dataframe_zero()
    else:
        batch_index += 1 # the value can be inherented
        ( x , y ) = datamanager.tail_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        scores = model.forward(x)
        _ , predict = torch.max(scores, 1) # predict is the first dimension , its the same as [ 1 ] 
        res = numpy.append( res, predict.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )  


    res = res[ :TESTING_INSTANCES ]
    res = res + 1

    numpy.savetxt( output_path, res, fmt='%d')

def save_elmo_rep_testset(model_dir, output_path, mtype='BiLSTMAttention'):
    '''
    Given Tokenized CharEncoded test file in TEST_SET_PATH, 
    save the embedded representation in output_path
    each line is label [sentence len] * [word embedding dim]
    '''
    model_path = '%s/model'%model_dir

    config = DefaultConfig() # Just take the default config to do the prediction work
    config.set_attrs( { 'batch_size' : 8 } )

    if mtype=='BiLSTMAttention':
        model=biLSTMAttention.BiLSTMAttention(
            param_document_seq_len = DOCUMENT_SEQ_LEN,# 300 in our model
            param_character_embedding_len = CHARACTER_EMBEDDING_LEN, #it depends on the setting
            param_bilstm_hidden_size = 1024 // 2, # 1024 is the Elmo size, the concatenated hidden size is supposed to Elmo size, however, any size is OK
            param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
            param_class_count = 5,
            param_options_file = config.options_file,
            param_weight_file = config.weight_file)
    print('Loading trained model')

    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            print('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    # here, load and save are defined in biLSTMAttention.py
    # load <=> model.load_state_dict( torch.load(path) )
    # save <=> torch.save( model.state_dict(), path )

    # an other way:
    # model = torch.load( path ) # has 2 field, if torch.save( model, path ), then both ['state_dict'] and ['struct'] != None
    # torch.save( model, path )

    if config.on_cuda:
        model.load( model_path )
        model=model.cuda()
    else:
        model.load_cpu_from_gputrained( model_path )
        model=model.cpu()
    
    # print(model)
    print('Begin loading data')
    datamanager=DataManager( param_batch_size = config.batch_size, param_training_instances_size = TESTING_INSTANCES) # the batch_size makes no differences
    datamanager.load_dataframe_from_file( TEST_SET_PATH )
    n_batch = datamanager.n_batches()
    res= numpy.empty( (0, DOCUMENT_SEQ_LEN * 1024), dtype = numpy.float32 ) # res is [], shape = (0, 3) , be sure to append on axis 0

    batch_index = 0

    for batch_index in range(n_batch - 1):
        ( x , y ) = datamanager.next_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        #elmo_rep = elmo_dict['elmo_representations']
        #var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = elmo_dict['elmo_representations'][0]
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )

    if TESTING_INSTANCES % config.batch_size == 0:
        datamanager.set_current_cursor_in_dataframe_zero()
    else:
        batch_index += 1 # the value can be inherented
        ( x , y ) = datamanager.tail_batch()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        #elmo_rep = elmo_dict['elmo_representations']
        #var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = elmo_dict['elmo_representations'][0]
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )  

    res = res[ :TESTING_INSTANCES ]

    numpy.savetxt( output_path, res, fmt='%f')    

def calculate_accuracy(model_dir):
    '''
    # actually, sklearn, meters.confusionMatrix and calculate_accuracy_and_recall.py can calculate the confusionMatrix
    '''
    fpInTestSet = open(TEST_SET_PATH, 'rt')
    fpInPredicted = open('%s/res.txt'%model_dir,'rt')
    rightPredictionCount = 0
    
    for alineG in fpInTestSet:
        (ground_truth , others) = alineG.strip().split( ' ', 1)
        alineP = fpInPredicted.readline()
        predicted = alineP.strip()
        if predicted == ground_truth:
            rightPredictionCount += 1
    fpInTestSet.close()
    fpInPredicted.close()
    
    acc = rightPredictionCount / TESTING_INSTANCES
    print(acc)

def save_elmo_rep(model_dir, input_path, output_path, mtype='BiLSTMAttention'):
    '''
    Given Tokenized CharEncoded txt file in input_path, 
    save the word embedded file in output_path
    each line is [sentence len] * [word embedding dim]
    '''
    model_path = '%s/model'%model_dir

    config = DefaultConfig() # Just take the default config to do the prediction work
    config.set_attrs( { 'batch_size' : 8 } )

    if mtype=='BiLSTMAttention':
        model=biLSTMAttention.BiLSTMAttention(
            param_document_seq_len = DOCUMENT_SEQ_LEN,# 300 in our model
            param_character_embedding_len = CHARACTER_EMBEDDING_LEN, #it depends on the setting
            param_bilstm_hidden_size = 1024 // 2, # 1024 is the Elmo size, the concatenated hidden size is supposed to Elmo size, however, any size is OK
            param_attention_size = (1024 // 2 * 2) // 1024 * 1024 + (1024 // 2 * 2) % 1024, # attention size should be a smoothed representation of character-emb
            param_class_count = 5,
            param_options_file = config.options_file,
            param_weight_file = config.weight_file)
    print('Loading trained model')

    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if config.on_cuda == False:
            print('Cuda is unavailable, Although wants to run on cuda, Model still run on CPU')

    # here, load and save are defined in biLSTMAttention.py
    # load <=> model.load_state_dict( torch.load(path) )
    # save <=> torch.save( model.state_dict(), path )

    # an other way:
    # model = torch.load( path ) # has 2 field, if torch.save( model, path ), then both ['state_dict'] and ['struct'] != None
    # torch.save( model, path )

    if config.on_cuda:
        model.load( model_path )
        model=model.cuda()
    else:
        model.load_cpu_from_gputrained( model_path )
        model=model.cpu()
    
    # print(model)
    print('Begin loading data')
    datamanager=DataManager( param_batch_size = config.batch_size, param_training_instances_size = TESTING_INSTANCES) # the batch_size makes no differences
    datamanager.load_dataframe_from_file( input_path )
    n_batch = datamanager.n_batches()
    res= numpy.empty( (0, DOCUMENT_SEQ_LEN * 1024), dtype = numpy.float32 ) # res is [], shape = (0, 3) , be sure to append on axis 0

    batch_index = 0

    for batch_index in range(n_batch - 1):
        x  = datamanager.next_batch_nolabel()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        #elmo_rep = elmo_dict['elmo_representations']
        #var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = elmo_dict['elmo_representations'][0]
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )

    if TESTING_INSTANCES % config.batch_size == 0:
        datamanager.set_current_cursor_in_dataframe_zero()
    else:
        batch_index += 1 # the value can be inherented
        x = datamanager.tail_batch_nolabel()
        x = Variable( torch.from_numpy(x).long(), requires_grad=False)
        if config.on_cuda:
            x = x.cuda()
        else:
            x = x.cpu()
        elmo_dict = model.forward_obtainTrainedElmoRep(x)
        #elmo_rep = elmo_dict['elmo_representations'][0]
        #var_elmo_rep = torch.cat( elmo_rep, dim = 0 ) # concatenate seq of tensors
        var_elmo_rep = elmo_dict['elmo_representations'][0]
        var_elmo_rep = var_elmo_rep.view(config.batch_size, DOCUMENT_SEQ_LEN * 1024 ) # 1024 is the Elmo size, fixed

        res = numpy.append( res, var_elmo_rep.data.cpu().numpy(), axis = 0 )

        print( '%d/%d'%( batch_index, n_batch ) )  

    res = res[ :TESTING_INSTANCES ]

    numpy.savetxt( output_path, res, fmt='%f')