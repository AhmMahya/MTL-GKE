function [Xs,Xs_label] = cv_Xs(X_src,Y_src)
    length_1=0;length_2=0;length_3=0;length_4=0;length_5=0;length_6=0;length_7=0;length_8=0;length_9=0;length_10=0;
    for j=1:size(X_src,2)
       if Y_src(j,1)==1
           length_1=length_1+1;
           Xs_1(length_1,:)=X_src(:,j)';
       elseif Y_src(j,1)==2
           length_2=length_2+1;
           Xs_2(length_2,:)=X_src(:,j)';
       elseif Y_src(j,1)==3
           length_3=length_3+1;
           Xs_3(length_3,:)=X_src(:,j)';
       elseif Y_src(j,1)==4
           length_4=length_4+1;
           Xs_4(length_4,:)=X_src(:,j)';
       elseif Y_src(j,1)==5
           length_5=length_5+1;
           Xs_5(length_5,:)=X_src(:,j)';
       elseif Y_src(j,1)==6
           length_6=length_6+1;
           Xs_6(length_6,:)=X_src(:,j)';
       elseif Y_src(j,1)==7
           length_7=length_7+1;
           Xs_7(length_7,:)=X_src(:,j)';
       elseif Y_src(j,1)==8
           length_8=length_8+1;
           Xs_8(length_8,:)=X_src(:,j)';
       elseif Y_src(j,1)==9
           length_9=length_9+1;
           Xs_9(length_9,:)=X_src(:,j)';
       elseif Y_src(j,1)==10
           length_10=length_10+1;
           Xs_10(length_10,:)=X_src(:,j)';
       end
    end


    %%%%%%%%%% seprate categories %%%%%%%%%%%%%


    ps=100;

    rand_index1=randperm(size(Xs_1,1),ceil(ps*length_1/100));
    data.data.train.source=Xs_1(rand_index1,:);
    data.labels.train.source=ones(1,ceil(ps*length_1/100));

    rand_index1=randperm(size(Xs_2,1),ceil(ps*length_2/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_2(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_2/100))+1);

    rand_index1=randperm(size(Xs_3,1),ceil(ps*length_3/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_3(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_3/100))+2);

    rand_index1=randperm(size(Xs_4,1),ceil(ps*length_4/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_4(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_4/100))+3);

    rand_index1=randperm(size(Xs_5,1),ceil(ps*length_5/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_5(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_5/100))+4);

    rand_index1=randperm(size(Xs_6,1),ceil(ps*length_6/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_6(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_6/100))+5);

    rand_index1=randperm(size(Xs_7,1),ceil(ps*length_7/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_7(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_7/100))+6);

    rand_index1=randperm(size(Xs_8,1),ceil(ps*length_8/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_8(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_8/100))+7);

    rand_index1=randperm(size(Xs_9,1),ceil(ps*length_9/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_9(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_9/100))+8);

    rand_index1=randperm(size(Xs_10,1),ceil(ps*length_10/100));
    data.data.train.source=cat(1,data.data.train.source,Xs_10(rand_index1,:));
    data.labels.train.source=cat(2,data.labels.train.source,ones(1,ceil(ps*length_10/100))+9);
    
    Xs = data.data.train.source;
    Xs_label = data.labels.train.source;

end