Search.setIndex({docnames:["index","lib","lib.modernisation","lib.ner_bert","lib.ner_lists","lib.post_correction","modules","server"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","lib.rst","lib.modernisation.rst","lib.ner_bert.rst","lib.ner_lists.rst","lib.post_correction.rst","modules.rst","server.rst"],objects:{"":{lib:[1,0,0,"-"],server:[7,0,0,"-"]},"lib.modernisation":{modernisation:[2,0,0,"-"],regex_rules:[2,0,0,"-"],syllable_corrector:[2,0,0,"-"],syllable_tokenizer:[2,0,0,"-"]},"lib.modernisation.modernisation":{Modernisation:[2,1,1,""]},"lib.modernisation.modernisation.Modernisation":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"lib.modernisation.regex_rules":{RegexRules:[2,1,1,""],split_line:[2,3,1,""]},"lib.modernisation.regex_rules.RegexRules":{__init__:[2,2,1,""],dict_lookup:[2,2,1,""],is_word_rule:[2,2,1,""],regex_subs:[2,2,1,""]},"lib.modernisation.syllable_corrector":{SyllableCorrector:[2,1,1,""]},"lib.modernisation.syllable_corrector.SyllableCorrector":{__call__:[2,2,1,""],__init__:[2,2,1,""]},"lib.modernisation.syllable_tokenizer":{Tokenizer:[2,1,1,""]},"lib.modernisation.syllable_tokenizer.Tokenizer":{__init__:[2,2,1,""],decode:[2,2,1,""],encode:[2,2,1,""],get_rules:[2,2,1,""],safety_check:[2,2,1,""],split_piece:[2,2,1,""],split_word:[2,2,1,""]},"lib.ner_bert":{inference:[3,0,0,"-"],ner_bert:[3,0,0,"-"]},"lib.ner_bert.inference":{Bert:[3,1,1,""]},"lib.ner_bert.inference.Bert":{__call__:[3,2,1,""],__init__:[3,2,1,""],create_bert:[3,2,1,""]},"lib.ner_bert.ner_bert":{MultipleBerts:[3,1,1,""]},"lib.ner_bert.ner_bert.MultipleBerts":{__call__:[3,2,1,""],__init__:[3,2,1,""],draw_conclusion:[3,2,1,""],map_bert_entity_result:[3,2,1,""],map_bert_results:[3,2,1,""],run_berts:[3,2,1,""]},"lib.ner_lists":{direct_list_creator:[4,0,0,"-"],entity:[4,0,0,"-"],find_entities_given_direct_list:[4,0,0,"-"],find_entities_given_tree:[4,0,0,"-"],finders:[4,0,0,"-"],get_searchable_words:[4,0,0,"-"],ner_lists:[4,0,0,"-"],tree_creator:[4,0,0,"-"],tree_matcher:[4,0,0,"-"]},"lib.ner_lists.direct_list_creator":{DirectListCreator:[4,1,1,""]},"lib.ner_lists.direct_list_creator.DirectListCreator":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"lib.ner_lists.entity":{Entity:[4,1,1,""]},"lib.ner_lists.entity.Entity":{__init__:[4,2,1,""]},"lib.ner_lists.find_entities_given_direct_list":{FindEntitiesGivenDirectList:[4,1,1,""]},"lib.ner_lists.find_entities_given_direct_list.FindEntitiesGivenDirectList":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"lib.ner_lists.find_entities_given_tree":{FindEntitiesGivenTree:[4,1,1,""]},"lib.ner_lists.find_entities_given_tree.FindEntitiesGivenTree":{__call__:[4,2,1,""],__init__:[4,2,1,""],remove_overlap:[4,2,1,""]},"lib.ner_lists.finders":{DirectListFinder:[4,1,1,""],Finder:[4,1,1,""],TreeFinder:[4,1,1,""]},"lib.ner_lists.finders.DirectListFinder":{__init__:[4,2,1,""]},"lib.ner_lists.finders.Finder":{__call__:[4,2,1,""],__init__:[4,2,1,""],merge_results:[4,2,1,""]},"lib.ner_lists.finders.TreeFinder":{__init__:[4,2,1,""]},"lib.ner_lists.get_searchable_words":{GetWords:[4,1,1,""],GetWordsBert:[4,1,1,""],GetWordsNer:[4,1,1,""]},"lib.ner_lists.get_searchable_words.GetWords":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"lib.ner_lists.get_searchable_words.GetWordsBert":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"lib.ner_lists.get_searchable_words.GetWordsNer":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"lib.ner_lists.ner_lists":{NerLists:[4,1,1,""]},"lib.ner_lists.ner_lists.NerLists":{__call__:[4,2,1,""],__init__:[4,2,1,""],prefill_empty_labels:[4,2,1,""]},"lib.ner_lists.tree_creator":{TreeCreator:[4,1,1,""],create_freq_table:[4,3,1,""],create_lists:[4,3,1,""],create_tree:[4,3,1,""]},"lib.ner_lists.tree_creator.TreeCreator":{__call__:[4,2,1,""],__init__:[4,2,1,""]},"lib.ner_lists.tree_matcher":{TreeMatcher:[4,1,1,""]},"lib.ner_lists.tree_matcher.TreeMatcher":{__init__:[4,2,1,""],find_matches:[4,2,1,""],get_close_matches_scores:[4,2,1,""],score:[4,2,1,""],yield_matches:[4,2,1,""]},"lib.pipeline":{Pipeline:[1,1,1,""]},"lib.pipeline.Pipeline":{__call__:[1,2,1,""],__init__:[1,2,1,""]},"lib.post_correction":{freq_table_clean_up:[5,0,0,"-"],post_correction:[5,0,0,"-"],replacer:[5,0,0,"-"],string_to_sentences:[5,0,0,"-"]},"lib.post_correction.freq_table_clean_up":{FreqTableCleanUp:[5,1,1,""]},"lib.post_correction.freq_table_clean_up.FreqTableCleanUp":{__call__:[5,2,1,""],__init__:[5,2,1,""]},"lib.post_correction.post_correction":{PostCorrection:[5,1,1,""]},"lib.post_correction.post_correction.PostCorrection":{__call__:[5,2,1,""],__init__:[5,2,1,""]},"lib.post_correction.replacer":{Replacer:[5,1,1,""]},"lib.post_correction.replacer.Replacer":{__init__:[5,2,1,""],get_replacement_inputs:[5,2,1,""],joinor:[5,2,1,""],replace_words:[5,2,1,""],return_matches:[5,2,1,""],splittor:[5,2,1,""]},"lib.post_correction.string_to_sentences":{StringToSentences:[5,1,1,""]},"lib.post_correction.string_to_sentences.StringToSentences":{__call__:[5,2,1,""],__init__:[5,2,1,""],get_list_of_dict_word_and_chars:[5,2,1,""],line_word_splits:[5,2,1,""],replacer:[5,4,1,""],set_ner_flag:[5,2,1,""],split_in_sentences:[5,2,1,""]},"lib.schema":{Validator:[1,1,1,""]},"lib.schema.Validator":{IGNORABLE:[1,4,1,""],__call__:[1,2,1,""],__init__:[1,2,1,""]},"server.exceptions":{BadRequest:[7,5,1,""],InternalServerError:[7,5,1,""]},"server.exceptions.BadRequest":{__init__:[7,2,1,""],status_code:[7,4,1,""],to_dict:[7,2,1,""]},"server.exceptions.InternalServerError":{__init__:[7,2,1,""],status_code:[7,4,1,""],to_dict:[7,2,1,""]},"server.flask_server":{call_pipeline:[7,3,1,""],handle_bad_request:[7,3,1,""],handle_internal_server_error:[7,3,1,""],main:[7,3,1,""]},lib:{constants:[1,0,0,"-"],modernisation:[2,0,0,"-"],ner_bert:[3,0,0,"-"],ner_lists:[4,0,0,"-"],pipeline:[1,0,0,"-"],post_correction:[5,0,0,"-"],schema:[1,0,0,"-"]},server:{exceptions:[7,0,0,"-"],flask_server:[7,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute","5":"py:exception"},terms:{"400":7,"500":7,"abstract":[3,4],"break":5,"char":5,"class":[1,2,3,4,5,7],"final":5,"function":[1,2,3,4,5],"return":[1,3,4,5,7],"static":[2,3,4,5],"throw":7,"true":[4,5],"while":4,CLS:2,For:5,The:[4,5,7],__call__:[1,2,3,4,5],__init__:[1,2,3,4,5,7],abc:3,accept:4,accord:1,accur:[1,2,3,4,5,7],add:2,affect:2,all:[4,7],all_label_prob:3,all_list:4,allow:1,also:[2,5],ani:[2,5],anti:5,appli:[3,5],applic:7,approxim:4,arbitrari:5,augment:7,backend:7,bad:7,badrequest:7,base:[1,2,3,4,5,7],bataviafort:4,becaus:4,befor:4,begin:[4,5],begin_char:5,belong:3,bert:[3,4],berts_to_initi:1,berts_to_us:3,best:4,better:5,between:4,bi_kei:3,bit:4,calcul:4,call:[1,2,3,4,5,7],call_pipelin:7,can:[4,5],canon:4,caus:[4,5],charact:[4,5],check:4,classmethod:3,clean:5,comma:4,compar:4,complic:4,concaten:4,consid:4,constant:[0,6],contain:[1,2,7],content:0,convert:5,correct:[1,5,7],create_bert:3,create_freq_t:4,create_list:4,create_tre:4,cutoff:4,cutoff_scor:4,data:[4,7],data_dir:4,deal:5,decod:2,definit:7,denot:7,depend:[1,4],detail:7,detect:5,dict:[1,2,4,5,7],dict_lookup:2,dictionari:[4,5],differ:[1,2,3],direct:7,direct_list:4,direct_list_cr:[1,6],directlistcr:4,directlistfind:4,divid:5,doing:4,draw_conclus:3,dutch:2,each:4,effici:4,either:[4,7],element:4,empti:[4,7],enabl:7,encod:2,end:[2,4,5],end_char:5,ent:4,entiti:[1,3,5,6],entity_kei:4,entity_typ:4,error:7,essenti:4,eventu:4,exampl:5,except:[0,6],execut:7,expect:[1,3],fallback:7,fals:[2,5],faster:4,file:2,file_nam:2,filepath:4,find:4,find_entities_given_direct_list:[1,6],find_entities_given_tre:[1,6],find_match:4,findentitiesgivendirectlist:4,findentitiesgiventre:4,finder:[1,6],first:[4,7],fix:5,flask:7,flask_serv:[0,6],follow:7,form:[5,7],format:[1,5,7],fortbatavia:4,found:4,found_ent:4,freq_table_clean_up:[1,6],freqtablecleanup:5,from:[2,4,7],from_kei:5,gener:5,get_close_matches_scor:4,get_list_of_dict_word_and_char:5,get_replacement_input:5,get_rul:2,get_searchable_word:[1,6],getword:4,getwordsbert:4,getwordsn:4,given:5,global:7,goal:2,group:4,handl:7,handle_bad_request:7,handle_internal_server_error:7,hello:5,help:[1,2,3,4,5,7],here:5,higher:4,histor:2,hit:4,i_sent:4,idempot:2,ignor:1,index:[0,4,5],indic:5,infer:[1,6],influenc:4,initi:[1,2,3,4,5,7],input:[1,7],input_str:[1,7],insid:5,integ:4,integr:1,interest:4,intermedi:4,intern:7,internalservererror:7,is_word_rul:2,joinor:5,json:[1,7],json_data:1,keep:4,kei:[4,5,7],kwarg:4,label_prob:3,larg:4,less:4,level:4,lib:[0,6],like:4,line:[2,5],line_word_split:5,linebreak:5,list:[2,3,4,5,7],list_nam:4,littl:4,load:[2,4,5],locat:[4,5],logic:1,look:5,low:4,made:5,main:7,make:2,map_bert_entity_result:3,map_bert_result:3,match:[4,5],max_dict:3,merg:4,merge_result:4,messag:7,method:[2,4,5,7],might:4,minim:4,miss:4,mistak:5,model:3,modern:2,modernis:[1,5,6,7],modul:[0,6],more:4,most:4,multipl:[3,4],multiplebert:3,n_approx:4,name:5,ner:[1,3,4,5],ner_bert:[1,6,7],ner_list:[1,6,7],ner_sent:3,nerlist:4,none:[1,2,3,4,7],normal:5,now:4,nr_tag:3,number:4,o_kei:3,object:[1,2,3,4,5],occur:4,ocr:5,old:2,onli:[2,4],option:7,order:7,origin:5,other:[4,5,7],output:1,over:5,packag:[0,6],pad:2,page:0,pair:4,paramet:4,part:1,partial:1,pattern:5,payload:7,per:[2,3,5],percentag:4,perform:4,permut:4,permutative_list:4,pipelin:[0,6,7],place:5,plain:7,point:7,poor:4,posit:4,possibl:[4,7],post:[1,5,7],post_correct:[1,6,7],postcorrect:5,prefill_empty_label:4,preselect:4,probabl:3,project:[1,4],provid:7,pyspellcheck:5,python:7,rais:7,realli:2,receiv:5,recognit:5,recurs:5,regex:5,regex_rul:[1,6],regex_sub:2,regexrul:2,remove_overlap:4,replac:[1,6],replace_funct:5,replace_word:5,repres:[4,5],request:7,requir:1,respon:5,respons:[1,2,3,5],result:[1,4,5,7],return_match:5,right:7,round:4,rule:2,run_bert:3,safety_check:2,schema:[0,6,7],score:4,search:0,searchabl:4,see:[1,2,3,4,5,7],self:[1,2,3,4,5,7],sentenc:[2,3,4,5],sep:2,server:[0,6],set:[2,4,5],set_ner_flag:5,shape:1,should:[4,5,7],signatur:[1,2,3,4,5,7],simpli:7,singl:4,size:4,small:[2,5],someth:4,somewhat:4,space:[4,5],special:5,specif:3,specifi:7,split:5,split_in_sent:5,split_lin:2,split_piec:2,split_word:2,splittor:5,start:[2,5],status_cod:7,step:[1,7],still:4,str:[2,5],string:[4,5,7],string_to_sent:[1,6],stringtosent:5,structur:7,sublist:5,submodul:[0,6],subpackag:[0,6],success:7,suitabl:5,syllable_corrector:[1,6],syllable_token:[1,6],syllablecorrector:2,table_nam:5,tag:2,tell:5,text:[4,5],thei:7,thi:[2,4,5,7],thing:5,threshold:4,time:4,to_dict:7,to_kei:5,token:2,too:4,top:4,total:4,translat:2,tree:4,tree_creat:[1,6],tree_match:[1,6],treecreat:4,treefind:4,treematch:4,tri:4,tupl:4,two:7,type:[1,2,3,4,5,7],type_of_list:4,unk:2,used:[3,5],user:7,using:2,valic:1,valid:1,verbos:2,verifi:1,vocab:3,want:4,webserv:7,weights_path:3,when:5,where:[5,7],which:[1,2,4,7],who:1,whole:[1,5],without:4,word:[2,3,4,5],word_form:5,word_form_lowercas:2,word_gett:4,word_group:4,word_list:4,word_piec:2,words_sub:5,yield_match:4,you:4},titles:["Welcome to ijsbeer-ai\u2019s documentation!","lib package","lib.modernisation package","lib.ner_bert package","lib.ner_lists package","lib.post_correction package","nationaal_archief_ner","server package"],titleterms:{constant:1,direct_list_cr:4,document:0,entiti:4,except:7,find_entities_given_direct_list:4,find_entities_given_tre:4,finder:4,flask_serv:7,freq_table_clean_up:5,get_searchable_word:4,ijsbeer:0,indic:0,infer:3,lib:[1,2,3,4,5],modernis:2,modul:[1,2,3,4,5,7],nationaal_archief_n:6,ner_bert:3,ner_list:4,packag:[1,2,3,4,5,7],pipelin:1,post_correct:5,regex_rul:2,replac:5,schema:1,server:7,string_to_sent:5,submodul:[1,2,3,4,5,7],subpackag:1,syllable_corrector:2,syllable_token:2,tabl:0,tree_creat:4,tree_match:4,welcom:0}})