import os
import threading
from shutil import copyfile
from ..utils import check_opts, logger_write, check_inputs,get_all_files,  split_list, run_shell




class kaldi_decoder:
    

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        

        # Setting logger and exec_config
        self.logger = logger
        
        self.output_folder=global_config['output_folder']

        # Definition of the expected options
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "decoding_script_folder": ("directory", "mandatory"),
            "decoding_script": ("file", "mandatory"),
            "graphdir": ("directory", "mandatory"),
            "alidir": ("directory", "mandatory"),
            "datadir": ("directory", "mandatory"),
            "posterior_folder": ("directory", "mandatory"),
            "save_folder": ("str", "optional","None"),
            "min_active": ("int(1,inf)", "optional","200"),
            "max_active": ("int(1,inf)", "optional","7000"),
            "max_mem": ("int(1,inf)", "optional","50000000"),
            "beam": ("float(0,inf)", "optional","13.0"),
            "lat_beam": ("float(0,inf)", "optional","8.0"),
            "acwt": ("float(0,inf)", "optional","0.2"),
            "max_arcs": ("int", "optional","-1"),
            "scoring": ("bool", "optional","True"),
            "scoring_script": ("file", "optional","None"),
            "scoring_opts": ("str", "optional","--min-lmwt 1 --max-lmwt 10"),
            "norm_vars": ("bool", "optional","False"),
            "data_text": ("directory", "optional","None"),
            "num_job": ("int(1,inf)", "optional","8"),
        }



        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling the class
        self.expected_inputs = [] 

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
        
        # Setting the save folder
        if self.save_folder is None:
            self.save_folder = self.output_folder + "/" + funct_name
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
                os.makedirs(self.save_folder+'/log')
            
        # getting absolute paths
        self.save_folder=os.path.abspath(self.save_folder)
        self.graphdir=os.path.abspath(self.graphdir)
        self.alidir=os.path.abspath(self.alidir)
        
        if self.scoring:
            self.scoring_script=os.path.abspath(self.scoring_script)
        # Reading all the ark files in the posterior_folder
        ark_files = get_all_files(
                self.posterior_folder,
                match_and=['.ark'],
            )
        
        # Sorting files (processing sentences of comparable size can make
        # multithreading much more efficient)
        ark_files.sort(key=lambda f: os.stat(f).st_size, reverse=False)

        # Deciding which sentences decoding in parallel
        N_chunks=int(len(ark_files)/self.num_job)
        ark_lists=split_list(ark_files,N_chunks)
        
        cnt=1
        
        # Manage multi-thread decoding
        for ark_list in ark_lists:
            
            threads = []
            
            for ark_file in ark_list:
           
                t = threading.Thread(target=self.decode_sentence, args=(ark_file,cnt))
                threads.append(t)
                t.start()                

                # Updating the sentence counter
                cnt=cnt+1
         
            for t in threads:
                t.join()
                
        
        if self.scoring:
            
            # copy final model as expectd by the kaldi scoring algorithm
            copyfile(self.alidir+'/final.mdl', self.output_folder+'/final.mdl')
            
            scoring_cmd= ('cd tools/kaldi_decoder/; %s %s %s %s %s' %(self.scoring_script, self.scoring_opts, self.datadir, self.graphdir, self.save_folder))
            
            # Running the scoring command
            run_shell(scoring_cmd,logger=self.logger)
            
            # Print scoring results()
            self.print_results()
            


                                  

    def __call__(self, inp_lst):
        return 
    
    
    def decode_sentence(self,ark_file,cnt):
        
        # Getting the absolute path
        ark_file=os.path.abspath(ark_file)
                
        # Composing the decoding command
        dec_cmd = (
                  'latgen-faster-mapped --min-active=%i --max-active=%i '
                  '--max-mem=%i --beam=%f --lattice-beam=%f '
                  '--acoustic-scale=%f --allow-partial=true '
                  '--word-symbol-table=%s/words.txt %s/final.mdl %s/HCLG.fst '
                  '"ark,s,cs: cat %s |" '
                  '"ark:|gzip -c > %s/lat.%i.gz"'
                   % (self.min_active, self.max_active, self.max_mem,
                      self.beam, self.lat_beam,self.acwt, self.graphdir, 
                      self.alidir, self.graphdir,ark_file,
                      self.save_folder,cnt)
                  ) 
        
        # Running the command
        run_shell(dec_cmd,logger=self.logger)
    
        
    
    def print_results(self):
        
        # Print the results (change it for librispeech scoring)
        subfolders = [f.path for f in os.scandir(self.save_folder) if f.is_dir() ]
        
        
        errors=[]
        
        for subfolder in subfolders:
            if 'score_' in  subfolder:
                files = os.listdir(subfolder)
                
                for file in files:
                    if '.sys' in file:
                        with open(subfolder+'/'+file) as fp:
                           line = fp.readline()
                           cnt = 1
                           while line:
                               if 'SPKR' in line and cnt==1:
                                   logger_write(line, logfile=self.logger,level='info')
                               if 'Mean' in line:
                                   line=line.replace(' Mean ',subfolder.split('/')[-1])
                                   logger_write(line, logfile=self.logger,level='info')
                                   
                                   line=line.replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
                                   errors.append(float(line.split('|')[-3].split(' ')[-3]))
                                   
                               line = fp.readline()
                               cnt += 1
        print(errors)                       
        logger_write('\nBEST ERROR RATE: %f\n' %(min(errors)), logfile=self.logger,level='info')

        
        
        
        
     
        
