Search.setIndex({docnames:["_generated/lit_saint","architecture","augmentations","index","uncertainty"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["_generated/lit_saint.rst","architecture.rst","augmentations.rst","index.rst","uncertainty.rst"],objects:{"":{lit_saint:[0,0,0,"-"]},"lit_saint.augmentations":{cutmix:[0,1,1,""],get_random_index:[0,1,1,""],mixup:[0,1,1,""]},"lit_saint.config":{AttentionTypeEnum:[0,2,1,""],AugmentationConfig:[0,2,1,""],ConstrastiveConfig:[0,2,1,""],ConstrativeEnum:[0,2,1,""],CutMixConfig:[0,2,1,""],DenoisingConfig:[0,2,1,""],MixUpConfig:[0,2,1,""],NetworkConfig:[0,2,1,""],OptimizerConfig:[0,2,1,""],PreTrainConfig:[0,2,1,""],PreTrainTaskConfig:[0,2,1,""],ProjectionHeadStyleEnum:[0,2,1,""],SaintConfig:[0,2,1,""],TrainConfig:[0,2,1,""],TransformerConfig:[0,2,1,""]},"lit_saint.config.AttentionTypeEnum":{col:[0,3,1,""],colrow:[0,3,1,""],row:[0,3,1,""]},"lit_saint.config.AugmentationConfig":{cutmix:[0,3,1,""],mixup:[0,3,1,""]},"lit_saint.config.ConstrastiveConfig":{constrastive_type:[0,3,1,""],dropout:[0,3,1,""],nce_temp:[0,3,1,""],projhead_style:[0,3,1,""],weight:[0,3,1,""]},"lit_saint.config.ConstrativeEnum":{simsiam:[0,3,1,""],standard:[0,3,1,""]},"lit_saint.config.CutMixConfig":{lam:[0,3,1,""]},"lit_saint.config.DenoisingConfig":{dropout:[0,3,1,""],scale_dim_internal_sepmlp:[0,3,1,""],weight_cross_entropy:[0,3,1,""],weight_mse:[0,3,1,""]},"lit_saint.config.MixUpConfig":{lam:[0,3,1,""]},"lit_saint.config.NetworkConfig":{dropout_embed_continuous:[0,3,1,""],embedding_size:[0,3,1,""],internal_dimension_embed_continuous:[0,3,1,""],num_workers:[0,3,1,""],transformer:[0,3,1,""]},"lit_saint.config.OptimizerConfig":{learning_rate:[0,3,1,""],other_params:[0,3,1,""]},"lit_saint.config.PreTrainConfig":{aug:[0,3,1,""],epochs:[0,3,1,""],optimizer:[0,3,1,""],task:[0,3,1,""]},"lit_saint.config.PreTrainTaskConfig":{contrastive:[0,3,1,""],denoising:[0,3,1,""]},"lit_saint.config.ProjectionHeadStyleEnum":{different:[0,3,1,""],same:[0,3,1,""]},"lit_saint.config.SaintConfig":{network:[0,3,1,""],pretrain:[0,3,1,""],train:[0,3,1,""]},"lit_saint.config.TrainConfig":{epochs:[0,3,1,""],internal_dimension_output_layer:[0,3,1,""],mlpfory_dropout:[0,3,1,""],optimizer:[0,3,1,""]},"lit_saint.config.TransformerConfig":{attention_type:[0,3,1,""],depth:[0,3,1,""],dim_head:[0,3,1,""],dropout:[0,3,1,""],heads:[0,3,1,""],scale_dim_internal_col:[0,3,1,""],scale_dim_internal_row:[0,3,1,""]},"lit_saint.datamodule":{SaintDatamodule:[0,2,1,""]},"lit_saint.datamodule.SaintDatamodule":{NAN_LABEL:[0,3,1,""],predict_dataloader:[0,4,1,""],prep:[0,4,1,""],scaler_continuous_columns:[0,4,1,""],set_predict_set:[0,4,1,""],set_pretraining:[0,4,1,""],test_dataloader:[0,4,1,""],train_dataloader:[0,4,1,""],val_dataloader:[0,4,1,""]},"lit_saint.dataset":{SaintDataset:[0,2,1,""]},"lit_saint.model":{Saint:[0,2,1,""]},"lit_saint.model.Saint":{configure_optimizers:[0,4,1,""],forward:[0,4,1,""],predict_step:[0,4,1,""],pretraining_step:[0,4,1,""],set_mcdropout:[0,4,1,""],set_pretraining:[0,4,1,""],shared_step:[0,4,1,""],test_step:[0,4,1,""],training:[0,3,1,""],training_step:[0,4,1,""],validation_step:[0,4,1,""]},"lit_saint.modules":{Attention:[0,2,1,""],GEGLU:[0,2,1,""],PreNorm:[0,2,1,""],Residual:[0,2,1,""],RowColTransformer:[0,2,1,""],SepMLP:[0,2,1,""],SimpleMLP:[0,2,1,""]},"lit_saint.modules.Attention":{forward:[0,4,1,""],training:[0,3,1,""]},"lit_saint.modules.GEGLU":{forward:[0,4,1,""],training:[0,3,1,""]},"lit_saint.modules.PreNorm":{forward:[0,4,1,""],training:[0,3,1,""]},"lit_saint.modules.Residual":{forward:[0,4,1,""],training:[0,3,1,""]},"lit_saint.modules.RowColTransformer":{forward:[0,4,1,""],forward_col:[0,4,1,""],forward_row:[0,4,1,""],training:[0,3,1,""]},"lit_saint.modules.SepMLP":{forward:[0,4,1,""],training:[0,3,1,""]},"lit_saint.modules.SimpleMLP":{forward:[0,4,1,""],training:[0,3,1,""]},"lit_saint.trainer":{SaintTrainer:[0,2,1,""]},"lit_saint.trainer.SaintTrainer":{fit:[0,4,1,""],get_model_from_checkpoint:[0,4,1,""],predict:[0,4,1,""],prefit:[0,4,1,""]},lit_saint:{augmentations:[0,0,0,"-"],config:[0,0,0,"-"],datamodule:[0,0,0,"-"],dataset:[0,0,0,"-"],model:[0,0,0,"-"],modules:[0,0,0,"-"],trainer:[0,0,0,"-"],version:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:method"},terms:{"0":[0,2],"0001":0,"00028":0,"01":0,"02":0,"1":[0,2,3],"10":0,"100":0,"16":0,"1704":0,"1e":0,"2":[0,1,2],"20":0,"2016":4,"3":[0,1,2],"4":0,"5":[0,2,3],"6":0,"64":0,"8":0,"99":0,"boolean":0,"case":[0,1,2],"class":0,"default":[0,3],"do":0,"enum":0,"final":1,"float":0,"function":[0,1,4],"import":3,"int":0,"new":3,"return":[0,1],"static":0,"true":[0,3],"while":0,A:0,And:0,At:0,But:0,By:0,If:0,In:[0,1,2,3,4],It:[0,1],The:0,Then:[1,3],There:[0,1],To:0,__class__:4,__name__:4,ab:0,about:4,abov:0,acc:0,acceler:0,accuraci:0,activ:[0,1],activation_modul:0,actual:0,ad:4,adam:0,add:[0,3],add_imag:0,addit:0,after:[0,1],afterward:0,algorithm:0,alia:0,all:[0,1,4],allow:[0,1],along:0,also:[0,1],although:0,an:[0,1,2,3],ani:0,anoth:2,anyth:0,anytim:4,append:[0,1],appli:[0,1,2,4],applic:[1,3],approach:1,approxim:4,ar:[0,1,2],architectur:3,arg:0,argmax:[0,3],argument:0,arxiv:[0,3],assign:3,associ:0,attent:[0,1],attention_typ:0,attentiontypeenum:0,attn:0,aug:0,augment:[1,3],augmentationconfig:0,automat:0,avail:[0,4],averag:[0,2],axi:3,back:0,backprop:0,backward:0,bar:0,base:0,base_config:3,basepredictionwrit:0,batch:0,batch_idx:0,bayesian:4,becom:4,been:0,befor:0,begin:3,being:0,below:0,best:0,between:[0,1,2],binari:0,bit:0,block:0,bool:0,both:[0,1],calcul:0,call:[0,4],callback:0,can:[0,1,3,4],care:0,carlo:3,cat_col:0,categor:[0,1],categori:[0,3],categorical_dim:3,cfg:3,checkpoint:0,choos:0,classif:[0,1],clever:4,closur:0,code:4,col:[0,1],colrow:0,column:[0,1,3],com:3,combin:[0,1,2],common:0,compon:3,comput:[0,1],con_col:0,condit:0,conf:3,config:3,configur:[0,3],configure_optim:0,connect:0,consid:1,consist:[0,1],constrast:0,constrastive_typ:0,constrastiveconfig:0,constrativeenum:0,contain:[0,3],continu:[0,1,3],continuo:0,contrast:[0,1],control:0,convert:0,core:0,correctli:3,correspond:0,cosineann:0,could:0,creat:[0,3],cross:1,current:0,custom:0,cutmix:[0,1,3],cutmixconfig:0,cycl:0,d:0,data:[0,1,3,4],data_loader_param:0,data_modul:3,databas:0,datafram:[0,3],dataload:0,dataloader_idx:0,datamodul:3,dataset:[2,3],ddp_spawn:0,decid:0,decod:0,deep:1,deepspe:0,def:0,defin:[0,3],definit:0,denois:[0,1],denoisingconfig:0,depth:0,describ:0,devic:0,df:[0,3],df_test:3,df_to_predict:3,dict:0,dictionari:0,differ:[0,1,2,4],dim:0,dim_head:0,dim_intern:0,dim_out:0,dim_out_for_each_feat:0,dim_target:[0,3],dimens:0,dis_opt:0,dis_sch:0,disabl:0,disk:0,displai:0,divid:0,dm:0,doe:4,don:0,drop:4,dropout:[0,1,3],dropout_embed_continu:0,dure:[0,4],e:0,each:[0,1,2],element:0,embed:[0,1],embedding_s:0,enabl:0,enable_pretrain:[0,3],encod:0,end:0,enforc:0,enhanc:1,entropi:1,enumer:0,epoch:0,error:0,estim:[0,3],eval:0,everi:0,exampl:0,example_imag:0,execut:0,experi:0,exponentiallr:0,f:3,factor:0,factori:0,fals:0,fancier:0,featur:[0,1],feed:0,ff:0,ff_dropout:0,fig:2,file:3,file_nam:3,find:[0,3],first:0,fit:[0,3],flag:0,fn:0,follow:3,former:0,forward:0,forward_col:0,forward_row:0,found:0,fp:3,framework:3,frequenc:0,from:[0,1,2,3,4],full:0,g:0,gal:4,gan:0,gate:0,gaussian:4,geglu:0,gelu:0,gen_opt:0,gen_sch:0,gener:[0,2],get:[0,1,4],get_model_from_checkpoint:0,get_random_index:0,ghahramani:4,github:3,given:[0,1],goe:0,gpu:0,gradient:0,grid:0,ground:4,ha:0,half:0,handl:0,happen:0,have:[0,2],head:0,here:0,hidden:0,hook:0,how:[0,4],http:[0,3],hydra:3,ignor:0,imag:[0,2],implement:[0,3],impli:0,improv:[0,4],includ:[0,1],independ:1,index:[0,3],indic:0,infer:0,initi:0,inplac:0,input:[0,1],instanc:[0,3],instead:[0,2],integ:0,interest:0,intern:0,internal_dimension_embed_continu:0,internal_dimension_output_lay:0,interpret:4,intersampl:0,interv:0,item:0,iter:0,its:[0,4],kei:0,keyword:0,know:0,known:4,kwarg:0,label:[0,1,3],labelencod:0,labels_hat:0,lam:[0,2],lambdalr:0,last:0,latter:0,layer:[0,1,4],lbfg:0,learn:[0,1],learning_r:0,learningratemonitor:0,len:0,lightn:0,lightningdatamodul:0,lightningmodul:[0,4],like:[0,3],line:3,linear:[0,1],list:0,lit:3,lit_saint:3,load:0,log:0,log_dict:0,logger:0,logic:0,logit:[0,1],loss:[0,1],loss_fn:0,lr:0,lr_dict:0,lr_schedul:0,lstm:0,m:4,made:0,make:[0,3],make_grid:0,mani:[0,4],mask:0,mathemat:4,max_epoch:3,mc_dropout:0,mc_dropout_iter:0,mcdropout:0,mean:0,mention:0,method:[0,1],metric:0,metric_to_track:0,metric_v:0,might:0,mixup:[0,1,3],mixupconfig:0,mlp:0,mlpfory_dropout:0,mode:0,model:[3,4],model_di:0,model_gen:0,modul:[3,4],monitor:0,mont:3,most:0,mse:1,multi:0,multipl:0,multipli:0,must:[0,1],mymodel:0,n_critic:0,name:[0,3],nan_label:0,nce_temp:0,ndarrai:0,need:[0,3],network:[0,3,4],networkconfig:0,neural:[0,1],neuron:4,never:0,next:0,nfeat:0,nn:0,nois:1,non:1,none:0,normal:0,note:0,np:3,num_work:0,number:0,numer:0,numerical_column:3,object:[0,1],obtain:[0,1],offici:3,often:[0,4],omegaconf:3,one:[0,1,3],ones:[1,3],onli:0,oom:0,open:3,oper:0,optim:0,optimizer1:0,optimizer2:0,optimizer_idx:0,optimizer_on:0,optimizer_step:0,optimizer_two:0,optimizerconfig:0,option:0,order:[0,1,3],ordinalencod:0,org:0,origin:[0,1,2],other:2,other_param:0,otherwis:0,out:[0,4],output:0,over:[0,1],overrid:0,overridden:0,own:0,packag:3,page:3,pair:2,paper:3,param:0,paramet:0,paramref:0,part:0,pass:0,patch:2,perform:[0,1,4],permut:[0,2],phase:0,pip:3,pixel:2,posit:2,possibl:0,pre:1,precict:0,precis:0,predict:[0,1,3,4],predict_dataload:0,predict_step:[0,4],predicts_step:0,prefit:0,prenorm:0,prep:0,preprocess:0,present:[0,2],pretra:0,pretrain:[0,3],pretrainconfig:0,pretraining_step:0,pretraintaskconfig:0,prevent:0,previou:0,probabilist:4,probabl:0,problem:[0,1],procedur:0,process:[0,4],produc:0,progress:0,project:0,projectionheadstyleenum:0,projhead_styl:0,propag:0,propos:4,provid:4,pseudocod:0,put:0,pytorch:[0,3],pytorch_lightn:0,random:[0,2],random_index:0,rate:0,real:2,realiz:4,reason:4,recip:0,reconstruct:[0,1],reducelronplateau:0,region:2,regist:0,regress:1,regular:4,relu:0,remov:2,replac:2,repo:3,repositori:3,repres:0,requir:0,residu:0,result:1,routin:0,row:[0,1,3],rowcol:1,rowcoltransform:[0,1],run:0,runtim:3,s:[0,1,4],saint:[0,1],saint_nan:0,saint_train:3,saintconfig:[0,3],saintdatamodul:[0,3],saintdataset:0,sainttrain:[0,3],same:[0,4],sampl:4,sample_img:0,save:[0,3],scale:0,scale_dim_intern:0,scale_dim_internal_col:0,scale_dim_internal_row:0,scale_dim_internal_sepmlp:0,scaler:0,scaler_continuous_column:0,scarc:1,schedul:0,scheduler1:0,scheduler2:0,scikit:0,scriptmodul:0,search:3,second:0,section:1,see:0,self:[0,1,4],selfattent:1,separ:[0,1],sepmlp:[0,1],sequenti:0,set:[0,3],set_mcdropout:0,set_predict_set:0,set_pretrain:0,sgd:0,share:0,shared_step:0,should:0,shown:0,silent:0,similar:[0,1],simplemlp:[0,1],simpli:[0,2,4],simplifi:0,simsiam:0,sinc:0,singl:0,size:0,skip:0,slice:0,smooth:0,so:[0,4],solv:1,some:[0,2],somepago:3,someth:0,sourc:0,space:4,spawn:0,specif:0,specifi:0,split:[0,3],split_column:[0,3],standard:0,standardscal:0,startswith:4,state:0,step:[0,3,4],stop:0,str:0,strategi:2,strict:0,style:0,su:0,subclass:0,submodul:3,subset:2,sum:0,supervis:1,support:0,sure:0,t:0,t_co:0,t_max:0,tabular:1,taht:0,take:0,taken:0,target:[0,1,3],target_categor:0,task:[0,1],techniqu:3,tell:0,temperatur:0,tensor:[0,2],test:[0,3,4],test_acc:0,test_batch:0,test_data:0,test_dataload:0,test_epoch_end:0,test_loss:0,test_out:0,test_step:0,text:0,thank:3,them:[0,1],thi:[0,1,2,3,4],thing:0,those:0,through:0,thu:0,time:[0,4],torch:0,torchvis:0,total:0,tpu:0,tpu_cor:0,train:[0,2,3,4],train_dataload:0,trainabl:4,trainconfig:0,trainer:3,training_step:0,tranform:0,transform:[0,1],transformerconfig:0,transformermixin:0,treat:4,tri:1,truncat:0,truncated_bptt_step:0,tupl:0,turn:4,two:0,type:[0,1,3],uncertainti:3,uniqu:0,unit:0,until:0,updat:0,us:[0,1,4],util:0,val:0,val_acc:0,val_batch:0,val_data:0,val_dataload:0,val_loss:0,val_out:0,valid:[0,3],validation_epoch_end:0,validation_step:0,validation_step_end:0,valu:[0,1,2,3],version:[2,3],w:3,wai:4,want:0,warn:0,wasserstein:0,we:[1,2,3,4],weight:[0,2],weight_cross_entropi:0,weight_ms:0,well:4,what:0,whatev:0,when:[0,1],where:[1,3],wherea:0,which:0,whose:0,within:0,without:0,won:0,work:4,would:3,write:0,x:0,x_categ:0,x_cont:0,y:0,you:[0,3],your:[0,3],z:0},titles:["lit_saint package","Network Architecture","Data Augmentation Techniques","Welcome to SAINT Lightning\u2019s documentation!","Uncertainty Estimation"],titleterms:{architectur:1,augment:[0,2],carlo:4,compon:1,config:0,content:[0,3],credit:3,cutmix:2,data:2,datamodul:0,dataset:0,document:3,dropout:4,estim:4,gener:3,how:3,implement:4,indic:3,instal:3,lightn:3,lit_saint:0,mixup:2,model:0,modul:0,mont:4,network:1,packag:0,pretrain:1,s:3,saint:3,step:1,submodul:0,tabl:3,techniqu:2,train:1,trainer:0,uncertainti:4,us:3,version:0,welcom:3,yaml:3}})