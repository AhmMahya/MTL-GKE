function [Xt_train,Xt_train_label,Xt_test,Xt_test_label] = cv_Xt(X_tar,Y_tar)
    length_1=0;length_2=0;length_3=0;length_4=0;length_5=0;length_6=0;length_7=0;length_8=0;length_9=0;length_10=0;
    pt=35;
    for j=1:size(X_tar,2)
       if Y_tar(j,1)==1
           length_1=length_1+1;
            Xs_1(length_1,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==2
           length_2=length_2+1;
           Xs_2(length_2,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==3
           length_3=length_3+1;
           Xs_3(length_3,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==4
           length_4=length_4+1;
           Xs_4(length_4,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==5
           length_5=length_5+1;
           Xs_5(length_5,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==6
           length_6=length_6+1;
           Xs_6(length_6,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==7
           length_7=length_7+1;
           Xs_7(length_7,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==8
           length_8=length_8+1;
           Xs_8(length_8,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==9
           length_9=length_9+1;
           Xs_9(length_9,:)=X_tar(:,j)';
       elseif Y_tar(j,1)==10
           length_10=length_10+1;
           Xs_10(length_10,:)=X_tar(:,j)';
       end
    end
    
    
    data.data.train.target=Xs_1(1:ceil(pt*length_1/100),:);
    data.labels.train.target=ones(1,size(Xs_1(1:ceil(pt*length_1/100),:),1));
    data.data.test.target=Xs_1(ceil(pt*length_1/100)+1:end,:);
    data.labels.test.target=ones(1,size(Xs_1(ceil(pt*length_1/100)+1:end,:),1));
    
    
    data.data.train.target=cat(1,data.data.train.target,Xs_2(1:ceil(pt*length_2/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_2(1:ceil(pt*length_2/100),:),1))+1);
    data.data.test.target=cat(1,data.data.test.target,Xs_2(ceil(pt*length_2/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_2(ceil(pt*length_2/100)+1:end,:),1))+1);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_3(1:ceil(pt*length_3/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_3(1:ceil(pt*length_3/100),:),1))+2);
    data.data.test.target=cat(1,data.data.test.target,Xs_3(ceil(pt*length_3/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_3(ceil(pt*length_3/100)+1:end,:),1))+2);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_4(1:ceil(pt*length_4/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_4(1:ceil(pt*length_4/100),:),1))+3);
    data.data.test.target=cat(1,data.data.test.target,Xs_4(ceil(pt*length_4/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_4(ceil(pt*length_4/100)+1:end,:),1))+3);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_5(1:ceil(pt*length_5/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_5(1:ceil(pt*length_5/100),:),1))+4);
    data.data.test.target=cat(1,data.data.test.target,Xs_5(ceil(pt*length_5/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_5(ceil(pt*length_5/100)+1:end,:),1))+4);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_6(1:ceil(pt*length_6/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_6(1:ceil(pt*length_6/100),:),1))+5);
    data.data.test.target=cat(1,data.data.test.target,Xs_6(ceil(pt*length_6/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_6(ceil(pt*length_6/100)+1:end,:),1))+5);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_7(1:ceil(pt*length_7/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_7(1:ceil(pt*length_7/100),:),1))+6);
    data.data.test.target=cat(1,data.data.test.target,Xs_7(ceil(pt*length_7/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_7(ceil(pt*length_7/100)+1:end,:),1))+6);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_8(1:ceil(pt*length_8/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_8(1:ceil(pt*length_8/100),:),1))+7);
    data.data.test.target=cat(1,data.data.test.target,Xs_8(ceil(pt*length_8/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_8(ceil(pt*length_8/100)+1:end,:),1))+7);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_9(1:ceil(pt*length_9/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_9(1:ceil(pt*length_9/100),:),1))+8);
    data.data.test.target=cat(1,data.data.test.target,Xs_9(ceil(pt*length_9/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_9(ceil(pt*length_9/100)+1:end,:),1))+8);
    
    data.data.train.target=cat(1,data.data.train.target,Xs_10(1:ceil(pt*length_10/100),:));
    data.labels.train.target=cat(2,data.labels.train.target,ones(1,size(Xs_10(1:ceil(pt*length_10/100),:),1))+9);
    data.data.test.target=cat(1,data.data.test.target,Xs_10(ceil(pt*length_10/100)+1:end,:));
    data.labels.test.target=cat(2,data.labels.test.target,ones(1,size(Xs_10(ceil(pt*length_10/100)+1:end,:),1))+9); 
    
    Xt_train = data.data.train.target;
    Xt_train_label = data.labels.train.target;
    Xt_test = data.data.test.target;
    Xt_test_label = data.labels.test.target;

end