mossformer2_librimix_2spk = {
        'model_type': "mossformer2",
        'sample_rate': 8000,
        'config_name': "mossformer2-librimix-2spk",

        'encoder_kernel_size': 16, #stride is infered to be kernelsize//2
        'encoder_out_nchannels': 512,
        'encoder_in_nchannels': 1,
        
        'masknet_numspks': 2,
        'masknet_chunksize': 250, #K
        'masknet_numlayers': 1,
        'masknet_norm': "ln",
        'masknet_useextralinearlayer': False,
        'masknet_extraskipconnection': True,

        'intra_numlayers': 24,
        'intra_nhead': 8,
        'intra_dffn': 1024,
        'intra_dropout': 0,
        'intra_use_positional': True,
        'intra_norm_before': True,
}

mossformer2_wsj0mix_3spk = {
        'model_type': "mossformer2",
        'sample_rate': 8000,
        'config_name': "mossformer2-wsj0mix-3spk",

        'encoder_kernel_size': 16, #stride is infered to be kernelsize//2
        'encoder_out_nchannels': 512,
        'encoder_in_nchannels': 1,
        
        'masknet_numspks': 3,
        'masknet_chunksize': 250, #K
        'masknet_numlayers': 1,
        'masknet_norm': "ln",
        'masknet_useextralinearlayer': False,
        'masknet_extraskipconnection': True,
        
        'intra_numlayers': 24,
        'intra_nhead': 8,
        'intra_dffn': 1024,
        'intra_dropout': 0,
        'intra_use_positional': True,
        'intra_norm_before': True,
}

mossformer2_whamr_2spk = {
        'model_type': "mossformer2",
        'sample_rate': 8000,
        'config_name': "mossformer2-whamr-2spk",

        'encoder_kernel_size': 16, #stride is infered to be kernelsize//2
        'encoder_out_nchannels': 512,
        'encoder_in_nchannels': 1,
        
        'masknet_numspks': 2,
        'masknet_chunksize': 250, #K
        'masknet_numlayers': 1,
        'masknet_norm': "ln",
        'masknet_useextralinearlayer': False,
        'masknet_extraskipconnection': True,
        
        'intra_numlayers': 24,
        'intra_nhead': 8,
        'intra_dffn': 1024,
        'intra_dropout': 0,
        'intra_use_positional': True,
        'intra_norm_before': True,
}